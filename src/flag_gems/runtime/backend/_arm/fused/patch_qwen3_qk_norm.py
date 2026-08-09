"""Fuse Qwen3 decode Q/K RMSNorm into one ordinary-Triton launch.

Qwen3 invokes ``q_norm(q_proj(x))`` followed immediately by
``k_norm(k_proj(x))``.  Fused QKV codegen places the decode Q and K projection
rows next to each other.  One register-light kernel normalizes that combined
input into one combined output allocation, replacing two launches while
preserving the normal functional (out-of-place) RMSNorm contract.
"""

from __future__ import annotations

import logging
import types

import torch
import triton
import triton.language as tl

from ..profile_range import profile_range
from ..vector_config import REDUCTION_TILE

logger = logging.getLogger(__name__)

_PATCHED: dict[int, "_QKNormCoordinator"] = {}


@triton.jit(do_not_specialize=["eps"])
def _qk_rms_norm_contiguous_kernel(
    qk_ptr,
    row_weight_ptr,
    out_ptr,
    head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize adjacent Q then K rows into one adjacent output."""
    pid = tl.program_id(0)
    row = qk_ptr + pid * head_dim
    out_row = out_ptr + pid * head_dim
    # The tiny norm weights are expanded once at model setup to one row per
    # Q/K head.  This removes the remaining runtime Q-vs-K select from the
    # generated loop and keeps the LLVM object equivalent to one regular
    # contiguous RMSNorm row program.
    weight = row_weight_ptr + pid * head_dim

    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, head_dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        valid = offsets < head_dim
        x = tl.load(row + offsets, mask=valid, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / head_dim + eps)

    for start in range(0, head_dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        valid = offsets < head_dim
        x = tl.load(row + offsets, mask=valid, other=0.0).to(tl.float32)
        w = tl.load(weight + offsets, mask=valid, other=0.0)
        # Preserve Qwen3 eager semantics: normalization is rounded to BF16
        # before the BF16 weight multiplication and final BF16 store.
        y = (x * rrms).to(out_row.dtype.element_ty) * w
        tl.store(out_row + offsets, y, mask=valid)


class _QKNormCoordinator:
    def __init__(self, attention: torch.nn.Module):
        self.attention = attention
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.original_q = self.q_norm.forward
        self.original_k = self.k_norm.forward
        self.pending_q: torch.Tensor | None = None
        self.pending_q_output: torch.Tensor | None = None
        self.pending_output: torch.Tensor | None = None
        self.head_dim = int(self.q_norm.weight.numel())
        if int(self.k_norm.weight.numel()) != self.head_dim:
            raise ValueError("Q/K RMSNorm head dimensions differ")
        self.eps = float(self.q_norm.variance_epsilon)
        if float(self.k_norm.variance_epsilon) != self.eps:
            raise ValueError("Q/K RMSNorm epsilon values differ")
        qkv = getattr(attention, "_triton_qkv_coordinator", None)
        if qkv is None:
            raise ValueError("fused Q/K norm requires fused QKV layout")
        k_elements = int(qkv.logical_sizes[1])
        if k_elements % self.head_dim:
            raise ValueError("K projection is not an integral number of heads")
        self.expected_k_rows = k_elements // self.head_dim
        self.qk_row_weight: torch.Tensor | None = None
        self.qk_row_shape: tuple[int, int] | None = None

    def _row_weights(self, q_rows: int, k_rows: int) -> torch.Tensor:
        shape = (q_rows, k_rows)
        if self.qk_row_weight is None or self.qk_row_shape != shape:
            self.qk_row_weight = torch.cat(
                (
                    self.q_norm.weight.expand(q_rows, -1),
                    self.k_norm.weight.expand(k_rows, -1),
                ),
                dim=0,
            ).contiguous()
            self.qk_row_shape = shape
        return self.qk_row_weight

    def _eligible(self, value: torch.Tensor) -> bool:
        return (
            not torch.is_grad_enabled()
            and value.dtype == torch.bfloat16
            and value.dim() == 4
            and value.shape[0] == 1
            and value.shape[1] == 1
            and value.shape[-1] == self.head_dim
            and value.is_contiguous()
            and self.q_norm.weight.dtype == torch.bfloat16
            and self.k_norm.weight.dtype == torch.bfloat16
        )

    def _materialize_pending(self) -> None:
        if self.pending_q is None:
            return
        pending = self.pending_q
        pending_output = self.pending_q_output
        self.pending_q = None
        self.pending_q_output = None
        self.pending_output = None
        if pending_output is not None:
            pending_output.copy_(self.original_q(pending))

    def q_forward(self, value: torch.Tensor) -> torch.Tensor:
        qkv = getattr(self.attention, "_triton_qkv_coordinator", None)
        consume = getattr(qkv, "consume_pre_normalized_qk", None)
        if consume is not None and consume(0, value):
            self._materialize_pending()
            return value
        if not self._eligible(value):
            self._materialize_pending()
            return self.original_q(value)
        # A second Q without an intervening K is outside Qwen3's normal call
        # order.  Materialize the previous result before deferring this one.
        self._materialize_pending()
        q_rows = value.numel() // self.head_dim
        self.pending_output = torch.empty(
            (q_rows + self.expected_k_rows, self.head_dim),
            dtype=value.dtype,
        )
        self.pending_q = value
        self.pending_q_output = self.pending_output[:q_rows].reshape(
            value.shape
        )
        return self.pending_q_output

    def k_forward(self, value: torch.Tensor) -> torch.Tensor:
        qkv = getattr(self.attention, "_triton_qkv_coordinator", None)
        consume = getattr(qkv, "consume_pre_normalized_qk", None)
        if consume is not None and consume(1, value):
            self.pending_q = None
            self.pending_q_output = None
            self.pending_output = None
            return value
        q = self.pending_q
        if (
            q is None
            or not self._eligible(value)
            or q.shape[2] <= 0
            or value.shape[2] <= 0
        ):
            self._materialize_pending()
            return self.original_k(value)

        output = self.pending_output
        q_output = self.pending_q_output
        q_rows = q.numel() // self.head_dim
        k_rows = value.numel() // self.head_dim
        q_end = q.data_ptr() + q.numel() * q.element_size()
        if (
            output is None
            or q_output is None
            or k_rows != self.expected_k_rows
            or value.data_ptr() != q_end
        ):
            self._materialize_pending()
            return self.original_k(value)

        self.pending_q = None
        self.pending_q_output = None
        self.pending_output = None
        with profile_range("triton::qk_rms_norm"):
            _qk_rms_norm_contiguous_kernel[(q_rows + k_rows,)](
                q,
                self._row_weights(q_rows, k_rows),
                output,
                self.head_dim,
                self.eps,
                BLOCK_SIZE=REDUCTION_TILE,
                num_warps=1,
                num_stages=1,
            )
        return output[q_rows:].reshape(value.shape)


def patch_qwen3_qk_norm(model: torch.nn.Module) -> int:
    """Patch compatible Qwen3 attention instances; return the patch count."""
    patched = 0
    for module in model.modules():
        if id(module) in _PATCHED:
            continue
        if not all(
            hasattr(module, name)
            for name in (
                "q_norm",
                "k_norm",
                "_triton_qkv_coordinator",
            )
        ):
            continue
        try:
            coordinator = _QKNormCoordinator(module)
        except (AttributeError, ValueError):
            continue
        module._triton_qk_norm_coordinator = coordinator
        coordinator.q_norm.forward = types.MethodType(
            lambda _self, value, _coord=coordinator: _coord.q_forward(value),
            coordinator.q_norm,
        )
        coordinator.k_norm.forward = types.MethodType(
            lambda _self, value, _coord=coordinator: _coord.k_forward(value),
            coordinator.k_norm,
        )
        _PATCHED[id(module)] = coordinator
        patched += 1
    if patched:
        logger.info(
            "Patched %d Qwen3 attention modules with fused Q/K RMSNorm",
            patched,
        )
    return patched


def unpatch_qwen3_qk_norm(model: torch.nn.Module) -> int:
    restored = 0
    for module in model.modules():
        coordinator = _PATCHED.pop(id(module), None)
        if coordinator is None:
            continue
        coordinator._materialize_pending()
        coordinator.q_norm.forward = coordinator.original_q
        coordinator.k_norm.forward = coordinator.original_k
        if hasattr(module, "_triton_qk_norm_coordinator"):
            del module._triton_qk_norm_coordinator
        restored += 1
    return restored
