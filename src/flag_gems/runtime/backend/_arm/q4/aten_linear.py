"""ATen/KleidiAI W4A8 linear used as a controlled Q4 baseline.

This is deliberately separate from the ordinary-Triton production router.
It lets the Qwen end-to-end benchmark preserve identical model-level fusion
while replacing only the Q4 matrix backend with PyTorch's native dynamic
INT8-activation/INT4-weight operator.
"""

from __future__ import annotations

import torch

from ..profile_range import profile_range
from .linear import BLOCK_LENGTH, quantize_q4_0


def prepare_aten_weight(
    weight: torch.Tensor,
    *,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Quantize Q4_0 weights and prepack them for ATen's KleidiAI kernel."""
    if not torch.backends.kleidiai.is_available():
        raise RuntimeError("ATen KleidiAI Q4 backend is unavailable")
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("ATen Q4 weight must be a floating-point [N,K] tensor")
    if weight.device.type != "cpu":
        raise ValueError("ATen Q4 weight preparation supports CPU tensors only")
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError(f"ATen Q4 requires N%4=0 and K%32=0; got {(n, k)}")
    if chunk_rows <= 0:
        raise ValueError("ATen Q4 chunk_rows must be positive")

    # ATen accepts two unsigned nibbles per byte and a BF16 scale per K32
    # group.  Bound temporary FP32/INT8 storage for the vocabulary matrix;
    # the final opaque prepack still consumes the complete tensors once.
    packed_nibbles = torch.empty((n, k // 2), dtype=torch.uint8)
    scales = torch.empty((n, k // BLOCK_LENGTH), dtype=torch.bfloat16)
    for row_begin in range(0, n, chunk_rows):
        row_end = min(row_begin + chunk_rows, n)
        quantized, scale = quantize_q4_0(weight[row_begin:row_end])
        unsigned = (quantized.to(torch.int16) + 8).to(torch.uint8)
        packed_nibbles[row_begin:row_end].copy_(
            unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
        )
        scales[row_begin:row_end].copy_(scale.to(torch.bfloat16))

    return torch.ops.aten._dyn_quant_pack_4bit_weight(
        packed_nibbles,
        scales,
        None,
        BLOCK_LENGTH,
        k,
        n,
    )


def prepare_aten_grouped_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Prepack an unpacked signed INT4 checkpoint without requantization.

    PyTorch's groupwise KleidiAI ABI consumes unsigned nibble codes in
    ``[N, K / group_size, group_size / 2]`` order and one BF16 scale per
    group.  Compressed-tensors stores the same values as signed INT8
    containers, so adding eight and packing adjacent K values preserves the
    checkpoint exactly.
    """
    if not torch.backends.kleidiai.is_available():
        raise RuntimeError("ATen KleidiAI Q4 backend is unavailable")
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError("grouped ATen Q4 weight must be INT8 [N,K]")
    if weight.device.type != "cpu" or weight_scale.device.type != "cpu":
        raise ValueError("grouped ATen Q4 preparation supports CPU only")
    n, k = weight.shape
    if (
        group_size <= 0
        or group_size % BLOCK_LENGTH
        or k % group_size
        or n % 4
    ):
        raise ValueError(
            "grouped ATen Q4 requires N%4=0 and a K32-aligned group size"
        )
    expected_scale_shape = (n, k // group_size)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"grouped ATen Q4 scale must have shape {expected_scale_shape}, "
            f"got {tuple(weight_scale.shape)}"
        )
    if int(weight.min()) < -8 or int(weight.max()) > 7:
        raise ValueError("grouped ATen Q4 values must be signed nibbles")

    unsigned = (weight.to(torch.int16) + 8).to(torch.uint8)
    packed_nibbles = (
        unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    ).reshape(n, k // group_size, group_size // 2)
    return torch.ops.aten._dyn_quant_pack_4bit_weight(
        packed_nibbles.contiguous(),
        weight_scale.to(torch.bfloat16).contiguous(),
        None,
        group_size,
        k,
        n,
    )


class AtenQ4Linear(torch.nn.Module):
    """Inference-only ATen/KleidiAI dynamic W4A8 linear."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
        *,
        profile_name: str = "aten_q4::linear",
        group_size: int = BLOCK_LENGTH,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.profile_name = profile_name
        self.group_size = int(group_size)
        self.register_buffer(
            "packed_weight", packed_weight.contiguous(), persistent=True
        )
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
        profile_name: str = "aten_q4::linear",
        chunk_rows: int = 1024,
    ) -> "AtenQ4Linear":
        if weight.ndim != 2:
            raise ValueError("AtenQ4Linear requires a two-dimensional weight")
        out_features, in_features = weight.shape
        packed = prepare_aten_weight(weight.detach(), chunk_rows=chunk_rows)
        return cls(
            packed,
            in_features,
            out_features,
            bias,
            profile_name=profile_name,
            group_size=BLOCK_LENGTH,
        )

    @classmethod
    def from_grouped_int4(
        cls,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        group_size: int,
        bias: torch.Tensor | None = None,
        profile_name: str = "aten_q4::linear",
    ) -> "AtenQ4Linear":
        if weight.ndim != 2:
            raise ValueError("AtenQ4Linear requires a two-dimensional weight")
        out_features, in_features = weight.shape
        packed = prepare_aten_grouped_weight(
            weight.detach(), weight_scale.detach(), group_size=group_size
        )
        return cls(
            packed,
            in_features,
            out_features,
            bias,
            profile_name=profile_name,
            group_size=group_size,
        )

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        *,
        profile_name: str = "aten_q4::linear",
    ) -> "AtenQ4Linear":
        return cls.from_weight(
            linear.weight,
            linear.bias,
            profile_name=profile_name,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("AtenQ4Linear is an inference-only module")
        shape = value.shape
        # PyTorch 2.10's CIX build advertises BF16 in the operator diagnostic,
        # but the enabled KleidiAI implementation accepts FP32 activation only.
        # Keep both conversions inside the profiled native baseline.
        with profile_range(self.profile_name):
            value_2d = value.reshape(-1, self.in_features).float().contiguous()
            output = torch.ops.aten._dyn_quant_matmul_4bit(
                value_2d,
                self.packed_weight,
                self.group_size,
                self.in_features,
                self.out_features,
            )
            output = output.to(value.dtype).reshape(
                *shape[:-1], self.out_features
            )
            if self.bias is not None:
                output = output + self.bias
        return output


__all__ = [
    "AtenQ4Linear",
    "prepare_aten_grouped_weight",
    "prepare_aten_weight",
]
