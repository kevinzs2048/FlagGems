"""Experimental complete W8A8 Linear, with native A8 packing and Triton GEMM.

No model registration: keep the measured native implementation as the default.
Weights use the original per-channel INT8 codes and FP32 scales unchanged.
"""

from dataclasses import dataclass

import torch
import triton

from flag_gems.runtime.backend._arm.ops.image_cpu_candidates import pack_integer_panels

# Kernel ownership lives in FlagGems. Re-export only for assembly diagnostics.
from flag_gems.runtime.backend._arm.ops.w8a8_packed_sme2 import (
    _wide_linear_kernel as _wide_linear_kernel,
)
from flag_gems.runtime.backend._arm.ops.w8a8_packed_sme2 import (
    w8a8_packed_sme2,
)
from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import ops


@dataclass(frozen=True)
class PackedWeight:
    data: torch.Tensor
    sums: torch.Tensor
    scales: torch.Tensor
    n: int
    k: int


def pack_weight(codes, scales):
    if (
        codes.device.type != "cpu"
        or codes.dtype != torch.int8
        or codes.ndim != 2
        or codes.shape[0] <= 0
        or codes.shape[0] % 64
        or not 256 <= codes.shape[1] <= 32768
        or codes.shape[1] % 128
        or scales.device.type != "cpu"
        or scales.dtype != torch.float32
        or scales.shape != (codes.shape[0],)
        or not torch.isfinite(scales).all()
        or not (scales > 0).all()
    ):
        raise ValueError(
            "Expected CPU W8 [N,K], N%64=0, K%128=0, 256<=K<=32768 and positive FP32 scales"
        )
    if list(ops().layout()) != [16, 64, 4, 1]:
        raise RuntimeError("Native A8 panel ABI differs from the validated layout")
    return PackedWeight(
        pack_integer_panels(codes),
        codes.sum(dim=1, dtype=torch.int32),
        scales.contiguous(),
        codes.shape[0],
        codes.shape[1],
    )


def from_packed_activation(
    packed, weight, m, *, workers=18, chunk=1, tile_n=64, row_epilogue=False
):
    if (
        type(m) is not int
        or m < 0
        or type(workers) is not int
        or workers < 1
        or type(tile_n) is not int
        or tile_n not in (16, 32, 64)
        or type(chunk) is not int
        or chunk not in (1, 2, 4, 8, 16)
        or packed.device.type != "cpu"
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or not packed.is_contiguous()
        or packed.numel() != triton.cdiv(m, 16) * 16 * (weight.k + 8)
    ):
        raise ValueError("Invalid native A8 panel or launch configuration")
    return w8a8_packed_sme2(
        packed,
        weight.data,
        weight.sums,
        weight.scales,
        m,
        weight.n,
        weight.k,
        workers=workers,
        chunk=chunk,
        tile_n=tile_n,
        row_epilogue=row_epilogue,
    )


def linear(x, weight, bias=None, *, workers=18, chunk=1, tile_n=64, row_epilogue=False):
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] != weight.k
    ):
        raise ValueError("Expected CPU BF16 [M,K] activation")
    if bias is not None and (
        bias.device.type != "cpu"
        or bias.shape != (weight.n,)
        or bias.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Expected CPU BF16/FP32 bias [N]")
    packed = ops().quant_pack_bf16(x)
    out = from_packed_activation(
        packed,
        weight,
        x.shape[0],
        workers=workers,
        chunk=chunk,
        tile_n=tile_n,
        row_epilogue=row_epilogue,
    )
    if bias is not None:
        out = out + bias.float()
    return out.bfloat16()


__all__ = [
    "_wide_linear_kernel",
    "PackedWeight",
    "pack_weight",
    "from_packed_activation",
]
