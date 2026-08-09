"""Monkey-patch Qwen3_5RMSNormGated.forward with ordinary Triton, replacing
a 6-op ATen sequence (pow + mean + rsqrt + mul × 2 + silu*mul) and turning
M single-row calls into one OMP-parallel multi-row kernel.

Per Qwen3.5-2B GDN decode token, RMSNormGated is invoked at:
  - GDN per-head norm (each linear_attention layer): 1 call/layer × 6 GDN
    layers × 1 token = 6 calls/tok, each shape [M=num_v_heads, D=head_v_dim]
    typically [16, 128].

Reference formula:
    out = (x / rms(x)) * weight * silu(gate)

Decode (BF16, [M, D] last-dim contiguous, M aligned). Other shapes /
dtypes fall back to the original forward.
"""
import logging
import types

import torch
import triton
import triton.language as tl

from ..vector_config import REDUCTION_TILE

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit(do_not_specialize=["eps"])
def _rms_norm_gated_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    out_ptr,
    D: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row * D
    lanes = tl.arange(0, BLOCK_SIZE)
    sum_sq = tl.zeros((1,), tl.float32)
    for base in range(0, D, BLOCK_SIZE):
        idx = base + lanes
        mask = idx < D
        x = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0).to(
            tl.float32
        )
        sum_sq += tl.sum(x * x, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / D + eps)
    for base in range(0, D, BLOCK_SIZE):
        idx = base + lanes
        mask = idx < D
        x = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = tl.load(gate_ptr + row_base + idx, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        # HF rounds the normalized value to the input dtype before applying
        # weight, while SiLU(gate) and the remaining epilogue stay in FP32.
        normalized = (x * rrms).to(tl.bfloat16).to(tl.float32)
        silu = gate / (1.0 + tl.exp(-gate))
        tl.store(
            out_ptr + row_base + idx,
            normalized * weight * silu,
            mask=mask,
        )


def _patched_rmsnorm_gated_forward(self, hidden_states, gate=None):
    if (
        gate is not None
        and hidden_states.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and gate.is_contiguous()
        and hidden_states.shape == gate.shape
        and hidden_states.shape[-1] == self._triton_D
    ):
        shape = hidden_states.shape
        D = self._triton_D
        M = hidden_states.numel() // D
        x_flat = hidden_states.reshape(M, D).contiguous()
        g_flat = gate.reshape(M, D).contiguous()
        out = torch.empty_like(x_flat)
        _rms_norm_gated_kernel[(M,)](
            x_flat,
            g_flat,
            self.weight,
            out,
            D=D,
            eps=float(self.variance_epsilon),
            BLOCK_SIZE=REDUCTION_TILE,
            num_warps=1,
            num_stages=1,
        )
        return out.reshape(*shape)

    # Fallback: original forward
    return self._original_forward(hidden_states, gate)


def _get_qwen3_5_rmsnorm_gated_classes():
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNormGated"),
        ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5RMSNormGated"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextRMSNormGated"),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_rmsnorm_gated(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_gated_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) not in _PATCHED:
            D = mod.weight.shape[0]
            mod._triton_D = D
            mod._original_forward = mod.forward
            mod.forward = types.MethodType(_patched_rmsnorm_gated_forward, mod)
            _PATCHED.add(id(mod))
            n += 1
    if n > 0:
        logger.info(
            "Patched %d Qwen3.5 RMSNormGated modules with Triton codegen", n
        )
    return n


def unpatch_qwen3_5_rmsnorm_gated(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_gated_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) in _PATCHED:
            if hasattr(mod, "_original_forward"):
                mod.forward = mod._original_forward
                del mod._original_forward
                del mod._triton_D
            _PATCHED.discard(id(mod))
            n += 1
    return n
