"""End-to-end Qwen3 Q4 inference route built from ordinary Triton kernels.

The Q4 matrix path lives in :mod:`.linear`.  This module supplies the model
level transformations needed to avoid treating 197 independent ``nn.Linear``
calls as the final abstraction:

* Q/K/V share one activation pack and one Q4 projection;
* gate/up share one activation pack and one Q4 projection;
* decode SwiGLU materialization and the down-projection A8 pack share one
  compiler-visible rolled Triton program;
* O, down, and the optional vocabulary projection use the same Q4 router;
* embedding, normalization, RoPE, attention, and greedy argmax use the
  existing ordinary-Triton Arm implementations.

No kernel in this module calls TLE_raw or an external compute runtime.
Framework-only view operations and cache ownership remain Python/PyTorch
concerns; they are reported separately from generated compute coverage.
"""

from __future__ import annotations

import logging
import os
import types
import weakref
from collections.abc import Iterable

import torch
import triton
import triton.language as tl

from ..ops.silu_and_mul import _SWIGLU_TILE, _sleef_expf_u10_inline
from ..profile_range import profile_range
from ..vector_config import VECTOR_BITS
from .linear import (
    BLOCK_LENGTH,
    _decode_partitions,
    _g128_decode_unroll,
    linear_w4a8,
    linear_w4a8_add_rmsnorm,
    linear_w4a8_rmsnorm,
    linear_w4a8_rmsnorm_qk_norm,
    pack_rhs_qsi4c32p,
    prepare_weight,
)
from .kernels import (
    _q4_asymmetric_qparams_from_minmax,
    _q4_decode_asym_g128_sdot_kernel,
    _q4_store_token_asymmetric_bf16_values,
)


logger = logging.getLogger(__name__)
_BF16_LANES = VECTOR_BITS // 16
_USE_FUSED_INPUT_RMSNORM_QKV = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_RMSNORM_QKV", "1"
).lower() in {"1", "true", "on"}
_USE_FUSED_POST_ADD_RMSNORM_GATEUP = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_ADD_RMSNORM_GATEUP", "1"
).lower() in {"1", "true", "on"}
_USE_FUSED_QK_NORM_QKV = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_QK_NORM_QKV", "1"
).lower() in {"1", "true", "on"}
_USE_DECODE_ATTENTION_FASTPATH = os.getenv(
    "FLAGGEMS_ARM_Q4_DECODE_ATTENTION_FASTPATH", "1"
).lower() in {"1", "true", "on"}
_USE_NATIVE_G128_RHS = os.getenv(
    "FLAGGEMS_ARM_Q4_NATIVE_G128_RHS", "1"
).lower() in {"1", "true", "on"}
_USE_FUSED_SWIGLU_DOWN_PACK = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_SWIGLU_DOWN_PACK", "1"
).lower() in {"1", "true", "on"}
_USE_FUSED_DOWN_RESIDUAL = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_DOWN_RESIDUAL", "1"
).lower() in {"1", "true", "on"}


def set_fused_input_rmsnorm_qkv_enabled(enabled: bool) -> bool:
    """Set the same-process Qwen input-RMSNorm/QKV A/B route."""
    global _USE_FUSED_INPUT_RMSNORM_QKV
    previous = _USE_FUSED_INPUT_RMSNORM_QKV
    _USE_FUSED_INPUT_RMSNORM_QKV = bool(enabled)
    return previous


def set_fused_post_add_rmsnorm_gateup_enabled(enabled: bool) -> bool:
    """Set the same-process Qwen post-attention/Gate-Up A/B route."""
    global _USE_FUSED_POST_ADD_RMSNORM_GATEUP
    previous = _USE_FUSED_POST_ADD_RMSNORM_GATEUP
    _USE_FUSED_POST_ADD_RMSNORM_GATEUP = bool(enabled)
    return previous


def set_fused_qk_norm_qkv_enabled(enabled: bool) -> bool:
    """Set the same-process joined-QKV/QK-head-norm A/B route."""
    global _USE_FUSED_QK_NORM_QKV
    previous = _USE_FUSED_QK_NORM_QKV
    _USE_FUSED_QK_NORM_QKV = bool(enabled)
    return previous


def set_decode_attention_fastpath_enabled(enabled: bool) -> bool:
    """Set the Q4 decode attention metadata fast path for profiling A/B."""
    global _USE_DECODE_ATTENTION_FASTPATH
    previous = _USE_DECODE_ATTENTION_FASTPATH
    _USE_DECODE_ATTENTION_FASTPATH = bool(enabled)
    return previous


def set_fused_swiglu_down_pack_enabled(enabled: bool) -> bool:
    """Set the same-process SwiGLU-to-Q4-down compact-pack A/B route."""
    global _USE_FUSED_SWIGLU_DOWN_PACK
    previous = _USE_FUSED_SWIGLU_DOWN_PACK
    _USE_FUSED_SWIGLU_DOWN_PACK = bool(enabled)
    return previous


def set_fused_down_residual_enabled(enabled: bool) -> bool:
    """Set the same-process Q4 down-store residual epilogue A/B route."""
    global _USE_FUSED_DOWN_RESIDUAL
    previous = _USE_FUSED_DOWN_RESIDUAL
    _USE_FUSED_DOWN_RESIDUAL = bool(enabled)
    return previous


@triton.jit
def _qwen3_embedding_rolled_kernel(
    indices_ptr,
    weight_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one selected embedding row with a native-width rolled loop."""
    row = tl.program_id(0)
    source_row = tl.load(indices_ptr + row)
    source = weight_ptr + source_row * N
    output = out_ptr + row * N
    lanes = tl.arange(0, BLOCK_SIZE)
    full_elements: tl.constexpr = (N // BLOCK_SIZE) * BLOCK_SIZE
    for base in tl.range(
        0, full_elements, BLOCK_SIZE, loop_unroll_factor=1
    ):
        values = tl.load(source + base + lanes)
        tl.store(output + base + lanes, values)
    if N % BLOCK_SIZE:
        offsets = full_elements + lanes
        valid = offsets < N
        values = tl.load(source + offsets, mask=valid, other=0.0)
        tl.store(output + offsets, values, mask=valid)


@triton.jit
def _qwen3_q4_swiglu_joined_kernel(
    joined_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Consume a contiguous ``[gate, up]`` row produced by fused Q4."""
    row = tl.program_id(0)
    joined = joined_ptr + row * (2 * N)
    output = out_ptr + row * N
    lanes = tl.arange(0, BLOCK_SIZE)
    full_elements: tl.constexpr = (N // BLOCK_SIZE) * BLOCK_SIZE
    for base in tl.range(
        0, full_elements, BLOCK_SIZE, loop_unroll_factor=1
    ):
        offsets = base + lanes
        gate = tl.load(joined + offsets).to(tl.float32)
        up = tl.load(joined + N + offsets).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        # Match Qwen's BF16 SiLU result before the following BF16 multiply.
        silu = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(output + offsets, (silu * up).to(tl.bfloat16))
    if N % BLOCK_SIZE:
        offsets = full_elements + lanes
        valid = offsets < N
        gate = tl.load(
            joined + offsets, mask=valid, other=0.0
        ).to(tl.float32)
        up = tl.load(
            joined + N + offsets, mask=valid, other=0.0
        ).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(
            output + offsets,
            (silu * up).to(tl.bfloat16),
            mask=valid,
        )


@triton.jit
def _qwen3_q4_swiglu_pack_asym_g128_kernel(
    joined_ptr,
    scratch_ptr,
    lhs_packed_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize BF16 SwiGLU and compact asymmetric A8 once per row."""
    row = tl.program_id(0)
    joined = joined_ptr + row * (2 * N)
    scratch = scratch_ptr + row * N
    packed = lhs_packed_ptr.to(tl.pointer_type(tl.uint8)) + row * (4 + N)
    lanes = tl.arange(0, BLOCK_SIZE)
    row_min = tl.full((1,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((1,), -3.4028234663852886e38, tl.float32)
    for base in tl.range(0, N, BLOCK_SIZE, loop_unroll_factor=1):
        offsets = base + lanes
        gate = tl.load(joined + offsets).to(tl.float32)
        up = tl.load(joined + N + offsets).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        value = (silu * up).to(tl.bfloat16)
        tl.store(scratch + offsets, value)
        value_f32 = value.to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(value_f32, axis=0))
        row_max = tl.maximum(row_max, tl.max(value_f32, axis=0))

    scale, inv_scale, zero_point = _q4_asymmetric_qparams_from_minmax(
        row_min, row_max
    )
    tl.store(
        packed.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 1), scale
    )
    tl.store(packed + 2 + tl.arange(0, 1), zero_point)
    lanes8 = tl.arange(0, 8)
    for group in tl.range(0, N // 32, loop_unroll_factor=1):
        source = scratch + group * 32
        _q4_store_token_asymmetric_bf16_values(
            packed + 4 + group * 32,
            tl.load(source + lanes8),
            tl.load(source + 8 + lanes8),
            tl.load(source + 16 + lanes8),
            tl.load(source + 24 + lanes8),
            inv_scale,
            zero_point,
        )


class Q4Linear(torch.nn.Module):
    """Inference-only linear backed by the ordinary-Triton Q4A8 router."""

    def __init__(
        self,
        rhs: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
        *,
        profile_name: str = "triton::q4_linear",
        activation_asymmetric: bool = False,
        weight_group_size: int = BLOCK_LENGTH,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.profile_name = profile_name
        self.activation_asymmetric = bool(activation_asymmetric)
        self.weight_group_size = int(weight_group_size)
        self.register_buffer("rhs", rhs.contiguous(), persistent=True)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(
                bias.detach().to(torch.bfloat16), requires_grad=False
            )

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        profile_name: str = "triton::q4_linear",
        chunk_rows: int = 1024,
    ) -> "Q4Linear":
        if weight.ndim != 2:
            raise ValueError("Q4Linear requires a two-dimensional weight")
        out_features, in_features = weight.shape
        rhs = prepare_weight(weight.detach(), chunk_rows=chunk_rows)
        return cls(
            rhs,
            in_features,
            out_features,
            bias,
            profile_name=profile_name,
        )

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        *,
        profile_name: str = "triton::q4_linear",
    ) -> "Q4Linear":
        return cls.from_weight(
            linear.weight,
            linear.bias,
            profile_name=profile_name,
        )

    @classmethod
    def from_grouped_int4(
        cls,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        group_size: int = 128,
        profile_name: str = "triton::q4_linear",
    ) -> "Q4Linear":
        """Pack an unpacked signed GPTQ tensor without requantization.

        Compressed-tensors stores signed INT4 values in INT8 containers and
        one dequantization scale per K group.  G128 checkpoints retain that
        native scale granularity in the production packed ABI instead of
        repeating metadata over four K32 dot panels.
        """
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("grouped Q4 weight must be INT8 [N,K]")
        n, k = weight.shape
        if group_size <= 0 or group_size % BLOCK_LENGTH or k % group_size:
            raise ValueError("Q4 group size must divide K and be K32-aligned")
        if n % 4 or int(weight.min()) < -8 or int(weight.max()) > 7:
            raise ValueError("grouped Q4 values must be signed nibbles")
        expected = (n, k // group_size)
        if tuple(weight_scale.shape) != expected:
            raise ValueError(
                f"grouped Q4 scale must have shape {expected}, "
                f"got {tuple(weight_scale.shape)}"
            )
        packed_group_size = (
            128 if group_size == 128 and _USE_NATIVE_G128_RHS
            else BLOCK_LENGTH
        )
        if packed_group_size == 128:
            from .linear import pack_rhs_qsi4c128p_asym

            rhs = pack_rhs_qsi4c128p_asym(
                weight.contiguous(), weight_scale.to(torch.bfloat16)
            )
        else:
            scales_k32 = weight_scale.to(torch.bfloat16).repeat_interleave(
                group_size // BLOCK_LENGTH, dim=1
            )
            from .linear import pack_rhs_qsi4c32p_asym

            rhs = pack_rhs_qsi4c32p_asym(weight.contiguous(), scales_k32)
        return cls(
            rhs,
            k,
            n,
            profile_name=profile_name,
            activation_asymmetric=True,
            weight_group_size=packed_group_size,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("Q4Linear is an inference-only module")
        with profile_range(self.profile_name):
            if self.activation_asymmetric:
                if self.weight_group_size == 128:
                    from .linear import linear_w4a8_asym_g128

                    function = linear_w4a8_asym_g128
                else:
                    from .linear import linear_w4a8_asym

                    function = linear_w4a8_asym
                output = function(
                    value, self.rhs, self.out_features, self.in_features
                )
            else:
                output = linear_w4a8(
                    value, self.rhs, self.out_features, self.in_features
                )
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward_rmsnorm(
        self,
        value: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
    ) -> torch.Tensor:
        """Decode projection with compiler-visible input RMSNorm fusion."""
        if self.bias is not None:
            raise ValueError("fused RMSNorm Q4 currently requires bias=False")
        if self.activation_asymmetric:
            if self.weight_group_size == 128:
                from .linear import linear_w4a8_asym_g128_rmsnorm

                function = linear_w4a8_asym_g128_rmsnorm
            else:
                from .linear import linear_w4a8_asym_rmsnorm

                function = linear_w4a8_asym_rmsnorm

            with profile_range("triton::q4_asym_rmsnorm_qkv"):
                return function(
                    value,
                    rms_weight,
                    rms_eps,
                    self.rhs,
                    self.out_features,
                    self.in_features,
                )
        with profile_range("triton::q4_rmsnorm_qkv"):
            return linear_w4a8_rmsnorm(
                value,
                rms_weight,
                rms_eps,
                self.rhs,
                self.out_features,
                self.in_features,
            )

    def forward_rmsnorm_qk_norm(
        self,
        value: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
        qk_weight: torch.Tensor,
        qk_eps: float,
        q_elements: int,
        k_elements: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Decode joined QKV with input and output-head normalization."""
        if self.bias is not None:
            raise ValueError("fused Q/K RMSNorm Q4 requires bias=False")
        with profile_range("triton::q4_rmsnorm_qkv_qk_norm"):
            return linear_w4a8_rmsnorm_qk_norm(
                value,
                rms_weight,
                rms_eps,
                qk_weight,
                qk_eps,
                q_elements,
                k_elements,
                head_dim,
                self.rhs,
                self.out_features,
                self.in_features,
            )

    def forward_add_rmsnorm(
        self,
        value: torch.Tensor,
        residual: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode projection with residual-add/RMSNorm fusion."""
        if self.bias is not None:
            raise ValueError("fused add-RMSNorm Q4 requires bias=False")
        if self.activation_asymmetric:
            if self.weight_group_size == 128:
                from .linear import linear_w4a8_asym_g128_add_rmsnorm

                function = linear_w4a8_asym_g128_add_rmsnorm
            else:
                from .linear import linear_w4a8_asym_add_rmsnorm

                function = linear_w4a8_asym_add_rmsnorm

            with profile_range("triton::q4_asym_add_rmsnorm_gate_up"):
                return function(
                    value,
                    residual,
                    rms_weight,
                    rms_eps,
                    self.rhs,
                    self.out_features,
                    self.in_features,
                )
        with profile_range("triton::q4_add_rmsnorm_gate_up"):
            return linear_w4a8_add_rmsnorm(
                value,
                residual,
                rms_weight,
                rms_eps,
                self.rhs,
                self.out_features,
                self.in_features,
            )


class TritonEmbedding(torch.nn.Module):
    """Inference embedding with a register-bounded ordinary-Triton copy."""

    def __init__(self, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        self.num_embeddings = int(embedding.num_embeddings)
        self.embedding_dim = int(embedding.embedding_dim)
        self.padding_idx = embedding.padding_idx
        self.weight = embedding.weight

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.device.type != "cpu" or self.weight.device.type != "cpu":
            raise ValueError("Arm Triton embedding supports CPU tensors only")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("embedding indices must be INT32 or INT64")
        flat = indices.contiguous().reshape(-1)
        output = torch.empty(
            (flat.numel(), self.embedding_dim), dtype=self.weight.dtype
        )
        with profile_range("triton::embedding"):
            _qwen3_embedding_rolled_kernel[(flat.numel(),)](
                flat,
                self.weight,
                output,
                N=self.embedding_dim,
                BLOCK_SIZE=_BF16_LANES,
                num_warps=1,
                num_stages=1,
            )
        return output.reshape(*indices.shape, self.embedding_dim)


class _Q4QKVCoordinator(torch.nn.Module):
    """Own one packed QKV matrix and serve Q, K, V calls as adjacent views."""

    def __init__(
        self,
        q_proj,
        k_proj,
        v_proj,
        *,
        linear_cls=Q4Linear,
        profile_prefix: str = "triton::",
    ) -> None:
        super().__init__()
        projections = (q_proj, k_proj, v_proj)
        if not all(isinstance(item, torch.nn.Linear) for item in projections):
            raise TypeError("Q/K/V fusion requires original nn.Linear modules")
        if any(item.bias is not None for item in projections):
            raise ValueError("Qwen3 Q/K/V fusion currently requires bias=False")
        if len({item.in_features for item in projections}) != 1:
            raise ValueError("Q/K/V input dimensions differ")
        self.logical_sizes = tuple(item.out_features for item in projections)
        self.in_features = int(q_proj.in_features)
        combined = torch.cat(
            tuple(item.weight.detach() for item in projections), dim=0
        )
        self.projection = linear_cls.from_weight(
            combined,
            profile_name=f"{profile_prefix}q4_qkv",
        )
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        self.cached_input: torch.Tensor | None = None
        self.cached_outputs: tuple[torch.Tensor, ...] | None = None
        self.pending_input_norm: (
            tuple[torch.Tensor, torch.Tensor, float] | None
        ) = None
        self.qk_norm_head_dim = 0
        self.qk_norm_eps = 0.0
        self.register_buffer("qk_norm_weight", None, persistent=False)
        self.qk_norm_enabled = False
        self.output_qk_normalized = False
        self.qk_norm_consume_index = 0

    @classmethod
    def from_projection(
        cls,
        projection: Q4Linear,
        logical_sizes: tuple[int, int, int],
        in_features: int,
    ) -> "_Q4QKVCoordinator":
        coordinator = cls.__new__(cls)
        torch.nn.Module.__init__(coordinator)
        coordinator.logical_sizes = tuple(int(item) for item in logical_sizes)
        coordinator.in_features = int(in_features)
        if sum(coordinator.logical_sizes) != projection.out_features:
            raise ValueError("joined QKV projection size does not match slices")
        if projection.in_features != coordinator.in_features:
            raise ValueError("joined QKV projection input size differs")
        coordinator.projection = projection
        coordinator._init_runtime_state()
        return coordinator

    def configure_qk_norm(
        self, q_norm: torch.nn.Module, k_norm: torch.nn.Module
    ) -> None:
        """Record the two per-head weights used by joined decode codegen."""
        head_dim = int(q_norm.weight.numel())
        if head_dim <= 0 or head_dim % 16:
            raise ValueError("Q/K norm head dimension must be 16-aligned")
        if int(k_norm.weight.numel()) != head_dim:
            raise ValueError("Q/K RMSNorm head dimensions differ")
        eps = float(q_norm.variance_epsilon)
        if float(k_norm.variance_epsilon) != eps:
            raise ValueError("Q/K RMSNorm epsilon values differ")
        if self.logical_sizes[0] % head_dim or self.logical_sizes[1] % head_dim:
            raise ValueError("Q/K projections must contain whole heads")
        self.qk_norm_head_dim = head_dim
        self.qk_norm_eps = eps
        self.qk_norm_weight = torch.cat(
            (q_norm.weight.detach(), k_norm.weight.detach()), dim=0
        ).to(torch.bfloat16).contiguous()
        self.qk_norm_enabled = True

    def consume_pre_normalized_qk(
        self, index: int, value: torch.Tensor
    ) -> bool:
        """Validate Q-then-K consumption of an already-normalized output."""
        if (
            not self.output_qk_normalized
            or index != self.qk_norm_consume_index
            or self.cached_outputs is None
        ):
            self.output_qk_normalized = False
            self.qk_norm_consume_index = 0
            return False
        expected = self.cached_outputs[index]
        if (
            value.device != expected.device
            or value.dtype != expected.dtype
            or value.numel() != expected.numel()
            or value.data_ptr() != expected.data_ptr()
        ):
            self.output_qk_normalized = False
            self.qk_norm_consume_index = 0
            return False
        self.qk_norm_consume_index += 1
        if index == 1:
            self.output_qk_normalized = False
            self.qk_norm_consume_index = 0
        return True

    def prepare_input_norm(
        self, value: torch.Tensor, norm: torch.nn.Module
    ) -> bool:
        """Defer an eligible decoder input norm into the Q4 projection."""
        if (
            not _USE_FUSED_INPUT_RMSNORM_QKV
            or not isinstance(self.projection, Q4Linear)
            or torch.is_grad_enabled()
            or value.device.type != "cpu"
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
            or value.shape[-1] != self.in_features
            or value.numel() // self.in_features >= 4
            or not hasattr(norm, "variance_epsilon")
            or norm.weight.device != value.device
            or norm.weight.dtype != torch.bfloat16
            or norm.weight.shape != (self.in_features,)
            or not norm.weight.is_contiguous()
        ):
            self.pending_input_norm = None
            return False
        self.pending_input_norm = (
            value,
            norm.weight,
            float(norm.variance_epsilon),
        )
        return True

    def _project_joined(self, value: torch.Tensor) -> torch.Tensor:
        pending = self.pending_input_norm
        self.pending_input_norm = None
        self.output_qk_normalized = False
        self.qk_norm_consume_index = 0
        if pending is not None and pending[0] is value:
            if (
                _USE_FUSED_QK_NORM_QKV
                and self.qk_norm_enabled
                and self.qk_norm_weight is not None
            ):
                joined = self.projection.forward_rmsnorm_qk_norm(
                    value,
                    pending[1],
                    pending[2],
                    self.qk_norm_weight,
                    self.qk_norm_eps,
                    self.logical_sizes[0],
                    self.logical_sizes[1],
                    self.qk_norm_head_dim,
                )
                self.output_qk_normalized = True
            else:
                joined = self.projection.forward_rmsnorm(
                    value, pending[1], pending[2]
                )
        else:
            joined = self.projection(value)
        return joined

    def _run(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        joined = self._project_joined(value)
        parts = torch.split(joined, self.logical_sizes, dim=-1)
        # For decode, the three views remain adjacent in one allocation.  The
        # Q/K norm fusion verifies this exact adjacency before using it.  For
        # prefill, each slice has a larger parent row stride and must be made
        # contiguous before Qwen reshapes it into heads.
        if value.numel() != value.shape[-1]:
            parts = tuple(part.contiguous() for part in parts)
        return tuple(parts)

    def can_project_joined_decode(self, value: torch.Tensor) -> bool:
        pending = self.pending_input_norm
        return (
            _USE_DECODE_ATTENTION_FASTPATH
            and (
                not self.qk_norm_enabled
                or (
                    _USE_FUSED_QK_NORM_QKV
                    and self.qk_norm_weight is not None
                )
            )
            and pending is not None
            and pending[0] is value
            and not torch.is_grad_enabled()
            and value.device.type == "cpu"
            and value.dtype == torch.bfloat16
            and value.dim() == 3
            and value.shape[0] == 1
            and value.shape[1] == 1
            and value.shape[2] == self.in_features
            and value.is_contiguous()
        )

    def project_joined_decode(self, value: torch.Tensor) -> torch.Tensor:
        """Return the one QKV allocation without split/module dispatches."""
        if not self.can_project_joined_decode(value):
            raise ValueError("joined Q4 decode fast path is not eligible")
        joined = self._project_joined(value)
        if self.qk_norm_enabled and not self.output_qk_normalized:
            raise RuntimeError("joined Q4 decode did not normalize Q/K heads")
        # No q_proj/k_proj/v_proj view modules will consume coordinator state.
        self.output_qk_normalized = False
        self.qk_norm_consume_index = 0
        self.cached_input = None
        self.cached_outputs = None
        return joined

    def project(self, index: int, value: torch.Tensor) -> torch.Tensor:
        if index == 0 or self.cached_input is not value:
            self.cached_outputs = self._run(value)
            self.cached_input = value
        assert self.cached_outputs is not None
        result = self.cached_outputs[index]
        if index == 2:
            self.cached_input = None
            self.cached_outputs = None
        return result


class _Q4ProjectionView(torch.nn.Module):
    """Non-owning module preserving Qwen's q_proj/k_proj/v_proj call sites."""

    def __init__(self, coordinator: _Q4QKVCoordinator, index: int) -> None:
        super().__init__()
        object.__setattr__(self, "_coordinator", weakref.ref(coordinator))
        self.index = int(index)
        self.in_features = coordinator.in_features
        self.out_features = coordinator.logical_sizes[index]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        coordinator = self._coordinator()
        if coordinator is None:
            raise RuntimeError("Q4 QKV coordinator has been released")
        return coordinator.project(self.index, value)


def _install_decode_attention_fastpath(
    attention: torch.nn.Module, coordinator: _Q4QKVCoordinator
) -> None:
    """Bypass Q/K/V module and split/view/transpose dispatches for decode."""
    original_forward = attention.forward
    original_globals = original_forward.__func__.__globals__
    all_attention_functions = original_globals["ALL_ATTENTION_FUNCTIONS"]
    eager_attention_forward = original_globals["eager_attention_forward"]

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
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
        total = int(joined.shape[-1])
        q_elements, k_elements, v_elements = coordinator.logical_sizes
        head_dim = coordinator.qk_norm_head_dim or int(attention.head_dim)
        q_heads = q_elements // head_dim
        k_heads = k_elements // head_dim
        base_offset = joined.storage_offset()
        # These are the exact post-transpose metadata strides expected by
        # attention.  Three as_strided views replace split_with_sizes plus
        # three view and three transpose dispatcher calls.
        query_states = joined.as_strided(
            (1, q_heads, 1, head_dim),
            (total, head_dim, total, 1),
            base_offset,
        )
        key_states = joined.as_strided(
            (1, k_heads, 1, head_dim),
            (total, head_dim, total, 1),
            base_offset + q_elements,
        )
        value_states = joined.as_strided(
            (1, k_heads, 1, head_dim),
            (total, head_dim, total, 1),
            base_offset + q_elements + k_elements,
        )
        if v_elements != k_elements:
            raise RuntimeError("Qwen3 K/V head layouts differ")

        cos, sin = position_embeddings
        # The class-level Arm RoPE patch is installed after model linears are
        # replaced.  Resolve the module global at call time so this instance
        # fast path observes that patch instead of retaining the eager
        # function object captured during setup.
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
        self._triton_q4_decode_fastpath_calls += 1
        return attn_output, attn_weights

    attention._triton_q4_original_forward = original_forward
    attention._triton_q4_decode_fastpath_calls = 0
    attention.forward = types.MethodType(decode_forward, attention)


class _Q4FusedMLP(torch.nn.Module):
    """Joined gate/up Q4 projection, Triton SwiGLU, and Q4 down projection."""

    def __init__(
        self,
        mlp: torch.nn.Module,
        *,
        linear_cls=Q4Linear,
        profile_prefix: str = "triton::",
    ) -> None:
        super().__init__()
        gate = mlp.gate_proj
        up = mlp.up_proj
        down = mlp.down_proj
        if not all(isinstance(item, torch.nn.Linear) for item in (gate, up, down)):
            raise TypeError("Q4 MLP fusion requires original nn.Linear modules")
        if any(item.bias is not None for item in (gate, up, down)):
            raise ValueError("Qwen3 Q4 MLP currently requires bias=False")
        if gate.in_features != up.in_features:
            raise ValueError("gate/up input dimensions differ")
        if gate.out_features != up.out_features:
            raise ValueError("gate/up output dimensions differ")
        if down.in_features != gate.out_features:
            raise ValueError("SwiGLU/down dimensions differ")
        self.intermediate_size = int(gate.out_features)
        gate_up = torch.cat((gate.weight.detach(), up.weight.detach()), dim=0)
        self.gate_up = linear_cls.from_weight(
            gate_up,
            profile_name=f"{profile_prefix}q4_gate_up",
        )
        self.down = linear_cls.from_linear(
            down,
            profile_name=f"{profile_prefix}q4_down",
        )

    @classmethod
    def from_projections(
        cls,
        gate_up: Q4Linear,
        down: Q4Linear,
        intermediate_size: int,
    ) -> "_Q4FusedMLP":
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module.intermediate_size = int(intermediate_size)
        if gate_up.out_features != 2 * module.intermediate_size:
            raise ValueError("joined gate/up output size differs")
        if down.in_features != module.intermediate_size:
            raise ValueError("down projection input size differs")
        module.gate_up = gate_up
        module.down = down
        return module

    def _can_fuse_swiglu_down_pack(self, joined: torch.Tensor, rows: int) -> bool:
        return (
            _USE_FUSED_SWIGLU_DOWN_PACK
            and rows < 4
            and isinstance(self.down, Q4Linear)
            and self.down.activation_asymmetric
            and self.down.weight_group_size == 128
            and self.down.bias is None
            and joined.device.type == "cpu"
            and joined.dtype == torch.bfloat16
            and joined.is_contiguous()
            and self.intermediate_size % 32 == 0
            and self.down.out_features % 4 == 0
        )

    def _finish_fused_swiglu_down_pack(
        self,
        joined: torch.Tensor,
        value_shape,
        rows: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activation = torch.empty(
            (*value_shape[:-1], self.intermediate_size), dtype=joined.dtype
        )
        lhs_packed = torch.empty(
            rows * (4 + self.intermediate_size), dtype=torch.uint8
        )
        with profile_range("triton::q4_swiglu_pack"):
            _qwen3_q4_swiglu_pack_asym_g128_kernel[(rows,)](
                joined,
                activation,
                lhs_packed,
                N=self.intermediate_size,
                BLOCK_SIZE=_SWIGLU_TILE,
                num_warps=1,
                num_stages=1,
            )
        output = torch.empty(
            (*value_shape[:-1], self.down.out_features), dtype=joined.dtype
        )
        partitions = _decode_partitions(
            self.down.in_features, self.down.out_features
        )
        with profile_range(self.down.profile_name):
            _q4_decode_asym_g128_sdot_kernel[(rows, partitions)](
                lhs_packed,
                self.down.rhs,
                output,
                output if residual is None else residual,
                0,
                self.down.out_features // 4,
                K=self.down.in_features,
                N=self.down.out_features,
                LHS_COMPACT=True,
                ADD_RESIDUAL=residual is not None,
                UNROLL=_g128_decode_unroll(self.down.in_features),
                num_warps=1,
                num_stages=1,
            )
        return output

    def _finish(self, joined: torch.Tensor, value_shape) -> torch.Tensor:
        rows = joined.numel() // (2 * self.intermediate_size)
        if self._can_fuse_swiglu_down_pack(joined, rows):
            return self._finish_fused_swiglu_down_pack(
                joined, value_shape, rows
            )
        activation = torch.empty(
            (*value_shape[:-1], self.intermediate_size), dtype=joined.dtype
        )
        with profile_range("triton::q4_swiglu"):
            _qwen3_q4_swiglu_joined_kernel[(rows,)](
                joined,
                activation,
                N=self.intermediate_size,
                BLOCK_SIZE=_SWIGLU_TILE,
                num_warps=1,
                num_stages=1,
            )
        return self.down(activation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._finish(self.gate_up(value), value.shape)

    def can_fuse_add_rmsnorm(
        self,
        value: torch.Tensor,
        residual: torch.Tensor,
        norm: torch.nn.Module,
    ) -> bool:
        return (
            _USE_FUSED_POST_ADD_RMSNORM_GATEUP
            and isinstance(self.gate_up, Q4Linear)
            and not torch.is_grad_enabled()
            and value.device.type == "cpu"
            and value.dtype == torch.bfloat16
            and value.shape == residual.shape
            and residual.dtype == value.dtype
            and value.shape[-1] == self.gate_up.in_features
            and value.numel() // value.shape[-1] < 4
            and hasattr(norm, "variance_epsilon")
            and norm.weight.device == value.device
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.shape == (self.gate_up.in_features,)
            and norm.weight.is_contiguous()
        )

    def forward_add_rmsnorm(
        self,
        value: torch.Tensor,
        residual: torch.Tensor,
        norm: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined, updated_residual = self.gate_up.forward_add_rmsnorm(
            value.contiguous(),
            residual.contiguous(),
            norm.weight,
            float(norm.variance_epsilon),
        )
        return self._finish(joined, value.shape), updated_residual

    def forward_add_rmsnorm_and_residual(
        self,
        value: torch.Tensor,
        residual: torch.Tensor,
        norm: torch.nn.Module,
    ) -> torch.Tensor:
        """Fuse post-attention norm, MLP, and final residual store epilogue."""
        joined, updated_residual = self.gate_up.forward_add_rmsnorm(
            value.contiguous(),
            residual.contiguous(),
            norm.weight,
            float(norm.variance_epsilon),
        )
        rows = joined.numel() // (2 * self.intermediate_size)
        if (
            _USE_FUSED_DOWN_RESIDUAL
            and self._can_fuse_swiglu_down_pack(joined, rows)
        ):
            return self._finish_fused_swiglu_down_pack(
                joined,
                value.shape,
                rows,
                residual=updated_residual,
            )
        return updated_residual + self._finish(joined, value.shape)


def _replace_attention_linears(
    attention: torch.nn.Module,
    *,
    linear_cls=Q4Linear,
    profile_prefix: str = "triton::",
    enable_qk_norm_fusion: bool = True,
) -> int:
    coordinator = _Q4QKVCoordinator(
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        linear_cls=linear_cls,
        profile_prefix=profile_prefix,
    )
    if (
        enable_qk_norm_fusion
        and _USE_FUSED_QK_NORM_QKV
        and linear_cls is Q4Linear
    ):
        coordinator.configure_qk_norm(attention.q_norm, attention.k_norm)
    attention._triton_qkv_coordinator = coordinator
    attention.q_proj = _Q4ProjectionView(coordinator, 0)
    attention.k_proj = _Q4ProjectionView(coordinator, 1)
    attention.v_proj = _Q4ProjectionView(coordinator, 2)
    attention.o_proj = linear_cls.from_linear(
        attention.o_proj,
        profile_name=f"{profile_prefix}q4_o_proj",
    )
    if (
        enable_qk_norm_fusion
        and _USE_DECODE_ATTENTION_FASTPATH
        and coordinator.qk_norm_enabled
        and linear_cls is Q4Linear
    ):
        _install_decode_attention_fastpath(attention, coordinator)
    return 4


def optimize_qwen3_q4(
    model: torch.nn.Module,
    *,
    skip_layers: Iterable[int] | None = None,
    quantize_lm_head: bool = True,
    enable_embedding: bool = False,
    enable_attention: bool = True,
    enable_argmax: bool = True,
    enable_qk_norm_fusion: bool = True,
) -> dict[str, object]:
    """Convert a Hugging Face Qwen3 model to the full ordinary-Triton Q4 path.

    Weight preparation is a one-time CPU operation.  The returned model is
    inference-only and its original BF16 Linear parameters are released as
    each decoder layer is converted.
    """
    return _optimize_qwen3_q4(
        model,
        skip_layers=skip_layers,
        quantize_lm_head=quantize_lm_head,
        enable_embedding=enable_embedding,
        enable_attention=enable_attention,
        enable_argmax=enable_argmax,
        enable_qk_norm_fusion=enable_qk_norm_fusion,
        linear_cls=Q4Linear,
        profile_prefix="triton::",
        quantization="live_grouped_q4a8_k32",
    )


def optimize_qwen3_q4_aten(
    model: torch.nn.Module,
    *,
    skip_layers: Iterable[int] | None = None,
    quantize_lm_head: bool = True,
    enable_embedding: bool = False,
    enable_attention: bool = True,
    enable_argmax: bool = True,
    enable_qk_norm_fusion: bool = True,
) -> dict[str, object]:
    """Use ATen/KleidiAI for Q4 matrices with identical model fusion."""
    from .aten_linear import AtenQ4Linear

    return _optimize_qwen3_q4(
        model,
        skip_layers=skip_layers,
        quantize_lm_head=quantize_lm_head,
        enable_embedding=enable_embedding,
        enable_attention=enable_attention,
        enable_argmax=enable_argmax,
        enable_qk_norm_fusion=enable_qk_norm_fusion,
        linear_cls=AtenQ4Linear,
        profile_prefix="aten_q4::",
        quantization="aten_kleidiai_dynamic_q4a8_k32",
    )


def _optimize_qwen3_q4(
    model: torch.nn.Module,
    *,
    skip_layers: Iterable[int] | None,
    quantize_lm_head: bool,
    enable_embedding: bool,
    enable_attention: bool,
    enable_argmax: bool,
    enable_qk_norm_fusion: bool,
    linear_cls,
    profile_prefix: str,
    quantization: str,
) -> dict[str, object]:
    if getattr(model, "_flag_gems_qwen3_q4_optimized", False):
        raise RuntimeError("Qwen3 model is already Q4 optimized")
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if core is None or layers is None:
        raise TypeError("optimize_qwen3_q4 expects Qwen3ForCausalLM")
    if torch.is_grad_enabled():
        logger.warning(
            "Q4 setup is inference-only; run forward under torch.inference_mode()"
        )

    skipped = set(skip_layers or ())
    converted_layers = 0
    logical_linears = 0
    for index, layer in enumerate(layers):
        if index in skipped:
            continue
        logical_linears += _replace_attention_linears(
            layer.self_attn,
            linear_cls=linear_cls,
            profile_prefix=profile_prefix,
            enable_qk_norm_fusion=enable_qk_norm_fusion,
        )
        layer.mlp = _Q4FusedMLP(
            layer.mlp,
            linear_cls=linear_cls,
            profile_prefix=profile_prefix,
        )
        logical_linears += 3
        converted_layers += 1

    if enable_embedding:
        if not isinstance(core.embed_tokens, torch.nn.Embedding):
            raise TypeError("Qwen3 embed_tokens is not nn.Embedding")
        core.embed_tokens = TritonEmbedding(core.embed_tokens)

    lm_head_converted = False
    if quantize_lm_head:
        lm_head = getattr(model, "lm_head", None)
        if not isinstance(lm_head, torch.nn.Linear):
            raise TypeError("Qwen3 lm_head is not nn.Linear")
        model.lm_head = linear_cls.from_linear(
            lm_head,
            profile_name=f"{profile_prefix}q4_lm_head",
        )
        logical_linears += 1
        lm_head_converted = True

    # Class-level patches retain unsupported-shape fallbacks, while the Q4
    # projection modules above own all model Linear computation.
    from ..fused.patch_qwen3_layer_norm import patch_qwen3_layer_norm
    from ..fused.patch_qwen3_rmsnorm import patch_qwen3_rmsnorm
    from ..fused.patch_qwen3_rope import patch_qwen3_rope

    # The standard Qwen3 frequency table is exact in BF16 and materially
    # faster than the eager matmul/cat/cos/sin sequence on CIX.
    os.environ.setdefault("FLAGGEMS_ARM_ROPE_FREQUENCY_CODEGEN", "1")
    patches = {
        "rmsnorm": patch_qwen3_rmsnorm(),
        "layer_norm": patch_qwen3_layer_norm(),
        "rope": patch_qwen3_rope(),
    }
    if enable_qk_norm_fusion:
        from ..fused.patch_qwen3_qk_norm import patch_qwen3_qk_norm

        patches["qk_norm_modules"] = patch_qwen3_qk_norm(model)

    overrides: list[str] = []
    if enable_attention:
        # Auto may otherwise select the legacy C runtime at short sequence
        # lengths.  The Q4 full-codegen route must stay compiler-visible.
        os.environ.setdefault("FLAGGEMS_ARM_ATTN_DISABLE_RUNTIME", "1")
        os.environ.setdefault(
            "FLAGGEMS_ARM_ATTN_SHORT_PREFILL_CODEGEN", "1"
        )
        overrides.append("scaled_dot_product_attention")
    if enable_argmax:
        from ..ops.argmax import set_argmax_vocab_assume_finite

        set_argmax_vocab_assume_finite(True)
        overrides.append("argmax")
    if overrides:
        from ..ops import apply_arm_overrides

        apply_arm_overrides(include=overrides)

    model._flag_gems_qwen3_q4_optimized = True
    return {
        "quantization": quantization,
        "decoder_layers": converted_layers,
        "logical_q4_linears": logical_linears,
        "physical_layer_matrices": converted_layers * 4,
        "physical_q4_matrices_total": (
            converted_layers * 4 + int(lm_head_converted)
        ),
        "lm_head_q4": lm_head_converted,
        "embedding_triton": enable_embedding,
        "qwen3_patches": patches,
        "input_rmsnorm_qkv_fusion": (
            _USE_FUSED_INPUT_RMSNORM_QKV and linear_cls is Q4Linear
        ),
        "post_add_rmsnorm_gateup_fusion": (
            _USE_FUSED_POST_ADD_RMSNORM_GATEUP and linear_cls is Q4Linear
        ),
        "qk_norm_qkv_fusion": (
            _USE_FUSED_QK_NORM_QKV
            and enable_qk_norm_fusion
            and linear_cls is Q4Linear
        ),
        "decode_attention_fastpath": (
            _USE_DECODE_ATTENTION_FASTPATH
            and _USE_FUSED_QK_NORM_QKV
            and enable_qk_norm_fusion
            and linear_cls is Q4Linear
        ),
        "arm_overrides": overrides,
    }


__all__ = [
    "Q4Linear",
    "TritonEmbedding",
    "optimize_qwen3_q4",
    "optimize_qwen3_q4_aten",
    "set_fused_input_rmsnorm_qkv_enabled",
    "set_fused_post_add_rmsnorm_gateup_enabled",
    "set_fused_qk_norm_qkv_enabled",
    "set_decode_attention_fastpath_enabled",
]
