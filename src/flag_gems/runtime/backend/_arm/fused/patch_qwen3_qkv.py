"""Fuse decode Q/K/V W8 projections into one compiler-generated SDOT matrix.

Qwen attention invokes ``q_proj(x)``, ``k_proj(x)``, and ``v_proj(x)`` in
that order with the same activation.  Three independent W8 decode kernels
repeat activation absmax/quantization and dispatch three independent matrix
kernels.  This patch concatenates their already block-major compiler packs
and coordinates the three module calls:

* q_proj quantizes once and computes all Q/K/V blocks in one matrix kernel;
* k_proj and v_proj return views cached by that launch;
* prefill and non-BF16 inputs retain the original three forwards.

The ordinary JIT path therefore uses two compiler-generated kernels
(quantizer + combined SDOT); its AOT wrapper presents the same pair as one
host call.  It never hides the matrix computation in a runtime helper.

The arithmetic for each output channel is unchanged.  This is a model-level
projection fusion over the existing compiler-visible SDOT lowering, not a
whole-operation runtime call.
"""

from __future__ import annotations

import logging
import os
import types

import torch

from ..int8.tle_int8_linear import (
    _quantize_bf16_w8_rne_kernel,
    _rmsnorm_quantize_bf16_w8_rne_kernel,
    _w8_decode_sdot_kernel,
    retile_weights_sdot_blocked,
)
from ..int8.aot_w8_backend import create_aot_w8_backend
from ..profile_range import profile_range

logger = logging.getLogger(__name__)

_PATCHED: set[int] = set()


class _QKVCoordinator:
    def __init__(self, q_proj, k_proj, v_proj):
        projections = (q_proj, k_proj, v_proj)
        required = (
            "_packed_codegen",
            "_w_scale_codegen",
            "_N_codegen",
            "_codegen_block_n",
            "K",
            "N",
        )
        if not all(all(hasattr(p, name) for name in required) for p in projections):
            raise TypeError("Q/K/V projections are not compiler-packed W8 linears")
        if len({p.K for p in projections}) != 1:
            raise ValueError("Q/K/V input dimensions differ")

        self.original = tuple(p.forward for p in projections)
        self.projections = projections
        self.k = q_proj.K
        self.block_ns = tuple(p._codegen_block_n for p in projections)
        whole_mode = os.getenv(
            "FLAGGEMS_ARM_QKV_WHOLE_CODEGEN", "auto"
        ).lower()
        self.whole_codegen = whole_mode in {"1", "true", "on"} or (
            whole_mode == "auto" and torch.get_num_threads() == 1
        )
        already_whole = all(
            getattr(p, "_whole_codegen", False) for p in projections
        )
        mixed_layout = any(
            getattr(p, "_whole_codegen", False) for p in projections
        ) and not already_whole
        if mixed_layout:
            raise ValueError("Q/K/V compiler packs use mixed layouts")
        if already_whole and len({p._whole_tile_n for p in projections}) != 1:
            raise ValueError("Q/K/V whole-codegen tile sizes differ")
        self.tile_n = q_proj._whole_tile_n if already_whole else 64
        if any(block_n % self.tile_n for block_n in self.block_ns):
            raise ValueError(
                f"Q/K/V block sizes {self.block_ns} are not divisible by "
                f"SDOT microtile {self.tile_n}"
            )
        self.logical_sizes = tuple(p.N for p in projections)
        self.padded_sizes = tuple(p._N_codegen for p in projections)
        self.n_codegen = sum(self.padded_sizes)
        if already_whole:
            packs = tuple(p._packed_codegen for p in projections)
        else:
            # Both the single-core whole kernel and the multicore shared-
            # quant grid consume the same spill-free N64 SDOT microtiles.
            packs = tuple(
                retile_weights_sdot_blocked(
                    p._packed_codegen, p._codegen_block_n, self.tile_n
                )
                for p in projections
            )
        self.packed = torch.cat(packs, dim=0).contiguous()
        self.scale = torch.cat(
            tuple(p._w_scale_codegen for p in projections), dim=0
        ).contiguous()
        self.ordinary_aot = (
            create_aot_w8_backend(self.k, self.n_codegen, self.tile_n)
            if self.whole_codegen
            else None
        )
        # The concatenated pack is the only decode pack used while patched.
        # Release the three source packs instead of retaining a second full
        # copy of every Q/K/V weight. Prefill uses each projection's separate
        # KAI N4/K8 pack, so its path remains intact.
        for projection in projections:
            projection._packed_codegen = None

        q_end = self.logical_sizes[0]
        k_start = self.padded_sizes[0]
        k_end = k_start + self.logical_sizes[1]
        v_start = self.padded_sizes[0] + self.padded_sizes[1]
        v_end = v_start + self.logical_sizes[2]
        self.slices = (
            slice(0, q_end),
            slice(k_start, k_end),
            slice(v_start, v_end),
        )
        self.offsets = (0, k_start, v_start)
        self.cached_input = None
        self.cached_outputs = None
        self.pending_input_norm = None

    @staticmethod
    def _is_decode(x: torch.Tensor) -> bool:
        return x.dtype == torch.bfloat16 and x.numel() == x.shape[-1]

    def _run_joined(self, x: torch.Tensor) -> torch.Tensor:
        flat = x if x.is_contiguous() else x.contiguous()
        pending_norm = self.pending_input_norm
        self.pending_input_norm = None
        output = torch.empty(self.n_codegen, dtype=torch.bfloat16)
        if self.ordinary_aot is not None:
            with profile_range("triton::w8_qkv_ordinary_aot"):
                self.ordinary_aot(flat, self.packed, self.scale, output)
        else:
            with profile_range("triton::w8_qkv_codegen"):
                x_q = torch.empty((self.k,), dtype=torch.int8)
                x_scale = torch.empty((1,), dtype=torch.float32)
                if pending_norm is not None and pending_norm[0] is x:
                    _rmsnorm_quantize_bf16_w8_rne_kernel[(1,)](
                        flat,
                        pending_norm[1],
                        x_q,
                        x_scale,
                        pending_norm[2],
                        K=self.k,
                        BLOCK_K=16,
                    )
                else:
                    _quantize_bf16_w8_rne_kernel[(1,)](
                        flat,
                        x_q,
                        x_scale,
                        K=self.k,
                        BLOCK_K=16,
                    )
                grid = (1,) if self.whole_codegen else (
                    self.n_codegen // self.tile_n,
                )
                _w8_decode_sdot_kernel[grid](
                    x_q,
                    x_scale,
                    self.packed,
                    self.scale,
                    output,
                    K=self.k,
                    N=self.n_codegen,
                    BLOCK_N=self.tile_n,
                    UNROLL=2,
                    WHOLE_PROJECTION=self.whole_codegen,
                )
        return output

    def prepare_input_norm(self, value, norm) -> bool:
        eligible = (
            not torch.is_grad_enabled()
            and self.ordinary_aot is None
            and self.can_project_joined_decode(value)
            and hasattr(norm, "variance_epsilon")
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.shape == (self.k,)
            and norm.weight.is_contiguous()
        )
        self.pending_input_norm = (
            value,
            norm.weight,
            float(norm.variance_epsilon),
        ) if eligible else None
        return eligible

    def _run(self, x: torch.Tensor):
        output = self._run_joined(x)
        prefix = x.shape[:-1]
        prefix_strides = tuple(self.n_codegen for _ in prefix)
        return tuple(
            output.as_strided(
                (*prefix, logical), (*prefix_strides, 1), offset
            )
            for offset, logical in zip(self.offsets, self.logical_sizes)
        )

    def can_project_joined_decode(self, x: torch.Tensor) -> bool:
        return (
            self._is_decode(x)
            and x.dim() == 3
            and x.shape[0] == 1
            and x.shape[1] == 1
            and x.is_contiguous()
        )

    def project_joined_decode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.can_project_joined_decode(x):
            raise ValueError("joined W8 decode fast path is not eligible")
        output = self._run_joined(x)
        self.cached_input = None
        self.cached_outputs = None
        return output

    def project(self, index: int, x: torch.Tensor):
        if not self._is_decode(x):
            return self.original[index](x)

        if index == 0:
            outputs = self._run(x)
            self.cached_input = x
            self.cached_outputs = outputs
            return outputs[0]

        if self.cached_input is not x or self.cached_outputs is None:
            # Preserve correctness if a model calls K/V without the normal
            # Q->K->V sequence.  It is still one fused projection launch.
            outputs = self._run(x)
            self.cached_input = x
            self.cached_outputs = outputs

        result = self.cached_outputs[index]
        if index == 2:
            self.cached_input = None
            self.cached_outputs = None
        return result


def _install_decode_attention_fastpath(attention, coordinator) -> None:
    """Consume the padded joined W8 allocation without split/view dispatches."""
    original_forward = attention.forward
    original_globals = original_forward.__func__.__globals__
    all_attention_functions = original_globals["ALL_ATTENTION_FUNCTIONS"]
    eager_attention_forward = original_globals["eager_attention_forward"]

    def decode_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        if not coordinator.can_project_joined_decode(hidden_states):
            return original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        joined = coordinator.project_joined_decode(hidden_states)
        q_elements, k_elements, v_elements = coordinator.logical_sizes
        q_offset, k_offset, v_offset = coordinator.offsets
        head_dim = int(self.head_dim)
        q_heads = q_elements // head_dim
        k_heads = k_elements // head_dim
        total = coordinator.n_codegen
        base = joined.storage_offset()
        query_states = joined.as_strided(
            (1, q_heads, 1, head_dim),
            (total, head_dim, total, 1),
            base + q_offset,
        )
        key_states = joined.as_strided(
            (1, k_heads, 1, head_dim),
            (total, head_dim, total, 1),
            base + k_offset,
        )
        value_states = joined.as_strided(
            (1, v_elements // head_dim, 1, head_dim),
            (total, head_dim, total, 1),
            base + v_offset,
        )
        cos, sin = position_embeddings
        query_states, key_states = original_globals["apply_rotary_pos_emb"](
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        attention_interface = all_attention_functions.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(1, 1, q_elements).contiguous()
        attn_output = self.o_proj(attn_output)
        self._triton_w8_decode_fastpath_calls += 1
        return attn_output, attn_weights

    attention._triton_w8_original_forward = original_forward
    attention._triton_w8_decode_fastpath_calls = 0
    attention.forward = types.MethodType(decode_forward, attention)


def patch_qwen3_qkv(model: torch.nn.Module) -> int:
    """Patch attention modules that own compatible W8 q/k/v projections."""
    patched = 0
    for module in model.modules():
        if id(module) in _PATCHED:
            continue
        if not all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue
        try:
            coordinator = _QKVCoordinator(
                module.q_proj, module.k_proj, module.v_proj
            )
        except (TypeError, ValueError):
            continue

        module._triton_qkv_coordinator = coordinator
        for index, projection in enumerate(coordinator.projections):
            projection.forward = types.MethodType(
                lambda _self, x, _index=index, _coord=coordinator:
                _coord.project(_index, x),
                projection,
            )
        # Llama-style attention has no post-projection Q/K RMSNorm.  Qwen3
        # retains its original forward until that normalization is fused too.
        if not hasattr(module, "q_norm") and hasattr(module, "head_dim"):
            _install_decode_attention_fastpath(module, coordinator)
        _PATCHED.add(id(module))
        patched += 1

    if patched:
        logger.info(
            "Patched %d attention modules with fused compiler-generated W8 QKV",
            patched,
        )
    return patched


def unpatch_qwen3_qkv(model: torch.nn.Module) -> int:
    restored = 0
    for module in model.modules():
        if id(module) not in _PATCHED or not hasattr(
            module, "_triton_qkv_coordinator"
        ):
            continue
        coordinator = module._triton_qkv_coordinator
        for projection, original in zip(
            coordinator.projections, coordinator.original
        ):
            if projection._packed_codegen is None:
                from ..int8.tle_int8_linear import (
                    pack_weights_sdot,
                    pack_weights_sdot_blocked,
                )

                projection._packed_codegen = pack_weights_sdot_blocked(
                    pack_weights_sdot(projection._get_w_int8_kn()),
                    projection._codegen_block_n,
                )
                projection._whole_codegen = False
                projection._whole_tile_n = None
            projection.forward = original
        del module._triton_qkv_coordinator
        _PATCHED.discard(id(module))
        restored += 1
    return restored
