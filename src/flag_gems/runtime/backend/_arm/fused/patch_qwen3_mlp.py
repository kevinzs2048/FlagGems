"""Monkey-patch Qwen3MLP.forward to use compiler-generated W8 SwiGLU stages.

This patch replaces the 5-op ATen sequence
    gate_proj(x) → silu → up_proj(x) → mul → down_proj
with a shared quantizer, one joined gate/up SDOT projection, and an ordinary
SwiGLU epilogue when:
    - decode shape (M=1)
    - BF16 activation
    - gate_proj and up_proj are INT8 SDOT-packed Linears
      (expose attributes: _packed, _w_scale, K, N)

Measured benefit on Qwen3-1.7B W8A8-INT8 decode (3 rounds × 5 runs median,
CIX P1 CD8180, 8 big cores, OMP=8, performance governor):

  ENABLE_MLP_PATCH=1  ON  → 9.92 tok/s median (9.88, 10.04, 9.92)
  ENABLE_MLP_PATCH=0  OFF → 9.73 tok/s median (9.61, 9.73, 9.76)
  → +1.95% median (+2.5% mean) consistent across 3 rounds.

Usage:
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp import patch_qwen3_mlp
    patch_qwen3_mlp(model)
"""

import logging
import os
import types

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cpu.tle_ops import fused_mlp as _tle_fused_mlp
except ImportError:
    _tle_fused_mlp = None

from ..int8.aot_w8_backend import create_aot_w8_mlp_backend
from ..int8.tle_int8_linear import (
    _add_rmsnorm_quantize_bf16_w8_rne_kernel,
    _quantize_bf16_w8_rne_kernel,
    _w8_decode_sdot_kernel,
    retile_weights_sdot_blocked,
)
from ..ops.silu_and_mul import (
    _SWIGLU_TILE,
    _swiglu_ordinary_joined_kernel,
    _swiglu_quantize_w8_rne_joined_kernel,
)
from ..profile_range import profile_range

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit
def _fused_mlp_kernel(
    x_ptr,
    gate_packed_ptr,
    up_packed_ptr,
    gate_scale_ptr,
    up_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    n_start = tl.program_id(0) * BLOCK_N
    _tle_fused_mlp(
        x_ptr,
        gate_packed_ptr,
        up_packed_ptr,
        gate_scale_ptr,
        up_scale_ptr,
        out_ptr,
        K,
        N,
        n_start,
        BLOCK_N,
    )


class FusedMLPWrapper:
    """Holds references to a Qwen3MLP's 3 projections + act_fn, exposes
    a forward that dispatches to compiler-generated SDOT on M=1 BF16 decode.

    The complete JIT MLP is deliberately four register-light kernels:
    input quantize, joined gate/up SDOT, SwiGLU plus down-input quantize, and
    down SDOT.  Combining dual SDOT accumulator banks and vector exp in one
    object increases Arm register pressure.  The AOT path retains its audited
    stage boundary and uses the regular down projection.

    Falls back to composing gate/up/down via their own forward for:
      - M > 1 (prefill)
      - non-BF16 activation
      - gate/up not SDOT-packed INT8 Linears
    """

    def __init__(self, gate_linear, up_linear, down_linear, act_fn):
        self._gate_linear = gate_linear
        self._up_linear = up_linear
        self.down_proj = down_linear
        self.act_fn = act_fn

        self._fused = (
            hasattr(gate_linear, "_packed_codegen")
            and hasattr(up_linear, "_packed_codegen")
            and hasattr(gate_linear, "_w_scale_codegen")
            and hasattr(up_linear, "_w_scale_codegen")
            and hasattr(gate_linear, "K")
            and hasattr(gate_linear, "N")
            and gate_linear._N_codegen == up_linear._N_codegen
            and gate_linear._codegen_block_n == up_linear._codegen_block_n
            and gate_linear._codegen_block_n % 32 == 0
            and gate_linear._packed_codegen is not None
            and up_linear._packed_codegen is not None
            and not getattr(gate_linear, "_whole_codegen", False)
            and not getattr(up_linear, "_whole_codegen", False)
        )
        self._ordinary_aot = None
        self._fused_down = False
        self._pending_input_quant = None
        if self._fused:
            self._K = gate_linear.K
            self._N = gate_linear.N
            self._N_codegen = gate_linear._N_codegen
            self._block_n = gate_linear._codegen_block_n
            self._ordinary_aot = create_aot_w8_mlp_backend(
                self._K, self._N_codegen, 64
            )
            self._packed_tile_n = 64
            self._gate_packed = retile_weights_sdot_blocked(
                gate_linear._packed_codegen,
                gate_linear._codegen_block_n,
                self._packed_tile_n,
            )
            self._up_packed = retile_weights_sdot_blocked(
                up_linear._packed_codegen,
                up_linear._codegen_block_n,
                self._packed_tile_n,
            )
            self._gate_scale = gate_linear._w_scale_codegen
            self._up_scale = up_linear._w_scale_codegen
            self._gate_up_packed = torch.cat(
                (self._gate_packed, self._up_packed), dim=0
            ).contiguous()
            self._gate_up_scale = torch.cat(
                (self._gate_scale, self._up_scale)
            ).contiguous()
            self._gate_packed = None
            self._up_packed = None
            # Decode uses the microtile packs above; prefill uses row-major
            # weights. Do not retain the superseded 512-output packs.
            gate_linear._packed_codegen = None
            up_linear._packed_codegen = None
            down_fusion_enabled = os.getenv(
                "FLAGGEMS_ARM_MLP_DOWN_QUANT_FUSION", "1"
            ).lower() not in {"0", "false", "off"}
            self._fused_down = (
                down_fusion_enabled
                and hasattr(down_linear, "_packed_codegen")
                and hasattr(down_linear, "_w_scale_codegen")
                and hasattr(down_linear, "_N_codegen")
                and hasattr(down_linear, "_codegen_block_n")
                and hasattr(down_linear, "K")
                and hasattr(down_linear, "N")
                and down_linear.K == self._N
                and down_linear._N_codegen % 64 == 0
                and down_linear._packed_codegen is not None
            )
            if self._fused_down:
                if not getattr(down_linear, "_whole_codegen", False):
                    down_linear._packed_codegen = retile_weights_sdot_blocked(
                        down_linear._packed_codegen,
                        down_linear._codegen_block_n,
                        64,
                    )
                    down_linear._whole_codegen = True
                    down_linear._whole_tile_n = 64
                elif down_linear._whole_tile_n != 64:
                    self._fused_down = False
            if self._fused_down:
                self._down_k = down_linear.K
                self._down_n = down_linear.N
                self._down_n_codegen = down_linear._N_codegen
                self._down_packed = down_linear._packed_codegen
                self._down_scale = down_linear._w_scale_codegen

    def forward(self, x):
        shape = x.shape
        M = x.numel() // shape[-1]
        if self._fused and M == 1 and x.dtype == torch.bfloat16:
            xc = x if x.is_contiguous() else x.contiguous()
            out = torch.empty(self._N_codegen, dtype=torch.bfloat16)
            if self._ordinary_aot is not None:
                with profile_range("triton::w8_mlp_ordinary_aot"):
                    self._ordinary_aot(
                        xc,
                        self._gate_up_packed,
                        self._gate_up_scale,
                        out,
                    )
            else:
                down_result = None
                with profile_range("triton::w8_mlp_ordinary_codegen"):
                    pending_quant = self._pending_input_quant
                    self._pending_input_quant = None
                    if pending_quant is None:
                        quantized = torch.empty((self._K,), dtype=torch.int8)
                        activation_scale = torch.empty(
                            (1,), dtype=torch.float32
                        )
                    else:
                        quantized, activation_scale = pending_quant
                    projection_out = torch.empty(
                        (2 * self._N_codegen,), dtype=torch.bfloat16
                    )
                    if pending_quant is None:
                        _quantize_bf16_w8_rne_kernel[(1,)](
                            xc,
                            quantized,
                            activation_scale,
                            K=self._K,
                            BLOCK_K=16,
                        )
                    whole_projection = torch.get_num_threads() == 1
                    grid = (1,) if whole_projection else (
                        2 * self._N_codegen // self._packed_tile_n,
                    )
                    _w8_decode_sdot_kernel[grid](
                        quantized,
                        activation_scale,
                        self._gate_up_packed,
                        self._gate_up_scale,
                        projection_out,
                        K=self._K,
                        N=2 * self._N_codegen,
                        BLOCK_N=self._packed_tile_n,
                        UNROLL=2,
                        WHOLE_PROJECTION=whole_projection,
                    )
                    if self._fused_down:
                        down_quantized = torch.empty(
                            (self._down_k,), dtype=torch.int8
                        )
                        down_activation_scale = torch.empty(
                            (1,), dtype=torch.float32
                        )
                        _swiglu_quantize_w8_rne_joined_kernel[(1,)](
                            projection_out,
                            out,
                            down_quantized,
                            down_activation_scale,
                            self._down_k,
                            BLOCK_SIZE=_SWIGLU_TILE,
                            num_warps=1,
                            num_stages=1,
                        )
                        down_result = torch.empty(
                            (*shape[:-1], self._down_n)
                            if self._down_n_codegen == self._down_n
                            else (self._down_n_codegen,),
                            dtype=torch.bfloat16,
                        )
                        down_whole = torch.get_num_threads() == 1
                        down_grid = (1,) if down_whole else (
                            self._down_n_codegen // 64,
                        )
                        _w8_decode_sdot_kernel[down_grid](
                            down_quantized,
                            down_activation_scale,
                            self._down_packed,
                            self._down_scale,
                            down_result,
                            K=self._down_k,
                            N=self._down_n_codegen,
                            BLOCK_N=64,
                            UNROLL=2,
                            WHOLE_PROJECTION=down_whole,
                        )
                    else:
                        _swiglu_ordinary_joined_kernel[(1,)](
                            projection_out,
                            out,
                            self._N_codegen,
                            BLOCK_SIZE=_SWIGLU_TILE,
                            num_warps=1,
                            num_stages=1,
                        )
                if down_result is not None:
                    if self._down_n_codegen == self._down_n:
                        return down_result
                    return down_result[: self._down_n].reshape(
                        *shape[:-1], self._down_n
                    )
            return self.down_proj(
                out[: self._N].reshape(*shape[:-1], self._N)
            )
        # ATen fallback: compose gate+up+silu+mul+down via each Linear's own forward
        gate = self._gate_linear(x)
        up = self._up_linear(x)
        return self.down_proj(self.act_fn(gate) * up)

    def can_fuse_add_rmsnorm(self, value, residual, norm) -> bool:
        return (
            self._fused
            and self._ordinary_aot is None
            and not torch.is_grad_enabled()
            and value.dtype == torch.bfloat16
            and value.shape == residual.shape
            and value.numel() == self._K
            and value.is_contiguous()
            and residual.is_contiguous()
            and hasattr(norm, "variance_epsilon")
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.shape == (self._K,)
            and norm.weight.is_contiguous()
        )

    def forward_add_rmsnorm(self, value, residual, norm):
        if not self.can_fuse_add_rmsnorm(value, residual, norm):
            raise ValueError("W8 add/RMSNorm/MLP fast path is not eligible")
        updated_residual = torch.empty_like(residual)
        quantized = torch.empty((self._K,), dtype=torch.int8)
        activation_scale = torch.empty((1,), dtype=torch.float32)
        with profile_range("triton::w8_add_rmsnorm_quantize"):
            _add_rmsnorm_quantize_bf16_w8_rne_kernel[(1,)](
                value,
                residual,
                norm.weight,
                updated_residual,
                quantized,
                activation_scale,
                float(norm.variance_epsilon),
                K=self._K,
                BLOCK_K=16,
                num_warps=1,
                num_stages=1,
            )
        self._pending_input_quant = (quantized, activation_scale)
        return self.forward(value), updated_residual


def _get_qwen_mlp_classes() -> tuple:
    """Return a tuple of MLP classes to patch (Qwen3MLP + Qwen3_5MLP if available).

    Both classes share the same structure (gate_proj, up_proj, down_proj, act_fn),
    so the FusedMLPWrapper works on either.
    """
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5MLP"),
        ("transformers.models.llama.modeling_llama", "LlamaMLP"),  # MiniCPM5 etc.
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_mlp(model) -> int:
    """Walk model, replace Qwen3MLP / Qwen3_5MLP forward with FusedMLPWrapper.

    Returns number of MLP instances patched. Safe to call multiple times —
    each instance is patched once (tracked via id).
    """
    mlp_classes = _get_qwen_mlp_classes()
    if not mlp_classes:
        logger.debug("No Qwen MLP classes found in transformers, skipping patch")
        return 0

    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, mlp_classes) and id(module) not in _PATCHED:
            wrapper = FusedMLPWrapper(
                module.gate_proj,
                module.up_proj,
                module.down_proj,
                module.act_fn,
            )
            module._original_forward = module.forward
            module._fused_mlp_wrapper = wrapper
            module.forward = types.MethodType(
                lambda self, x, _w=wrapper: _w.forward(x),
                module,
            )
            # The decoder-layer patch probes these methods on the original
            # MLP module.  Keep the wrapper private while exposing only the
            # two fusion hooks needed to consume add+RMSNorm directly as the
            # already-quantized input of the joined gate/up projection.
            module.can_fuse_add_rmsnorm = types.MethodType(
                lambda self, value, residual, norm, _w=wrapper: (
                    _w.can_fuse_add_rmsnorm(value, residual, norm)
                ),
                module,
            )
            module.forward_add_rmsnorm = types.MethodType(
                lambda self, value, residual, norm, _w=wrapper: (
                    _w.forward_add_rmsnorm(value, residual, norm)
                ),
                module,
            )
            _PATCHED.add(id(module))
            n += 1
    if n > 0:
        cls_names = ", ".join(c.__name__ for c in mlp_classes)
        logger.info(
            "Patched %d MLP modules (classes: %s) with fused_mlp_bf16", n, cls_names
        )
    return n


def unpatch_qwen3_mlp(model) -> int:
    """Restore original MLP forward (for testing / revert)."""
    mlp_classes = _get_qwen_mlp_classes()
    if not mlp_classes:
        return 0
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, mlp_classes) and id(module) in _PATCHED:
            if hasattr(module, "_original_forward"):
                wrapper = module._fused_mlp_wrapper
                from ..int8.tle_int8_linear import (
                    pack_weights_sdot,
                    pack_weights_sdot_blocked,
                )

                for projection in (
                    wrapper._gate_linear,
                    wrapper._up_linear,
                ):
                    if projection._packed_codegen is None:
                        projection._packed_codegen = (
                            pack_weights_sdot_blocked(
                                pack_weights_sdot(
                                    projection._get_w_int8_kn()
                                ),
                                projection._codegen_block_n,
                            )
                        )
                        projection._whole_codegen = False
                        projection._whole_tile_n = None
                module.forward = module._original_forward
                del module._original_forward
                del module._fused_mlp_wrapper
                del module.can_fuse_add_rmsnorm
                del module.forward_add_rmsnorm
            _PATCHED.discard(id(module))
            n += 1
    return n
