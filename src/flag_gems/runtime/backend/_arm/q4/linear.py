"""Production Q4A8 linear router for Arm Triton-CPU.

The prefill route uses ordinary Triton for both activation quantization and
the portable Q4 matrix multiply. Decode (M < 4) remains a dedicated
ordinary-Triton SDOT kernel because four-row I8MM is not an efficient one-row
kernel. No route calls a TLE/runtime matrix implementation.
"""

from __future__ import annotations

import logging
import math
import os
import platform

import torch

from .kernels import (
    _q4_fused_add_rmsnorm_decode_sdot_kai_kernel,
    _q4_fused_decode_sdot_kai_kernel,
    _q4_fused_rmsnorm_decode_sdot_kai_kernel,
    _q4_fused_rmsnorm_qk_norm_decode_sdot_kai_kernel,
    _pack_lhs_qsi8d32p_asym_decode_kernel,
    _pack_lhs_qsi8d32p_asym_add_rmsnorm_decode_kernel,
    _pack_lhs_qsi8d32p_asym_rmsnorm_decode_kernel,
    _pack_lhs_qsi8d32p_decode_kernel,
    _pack_lhs_qsi8d128p_asym_panel4_kernel,
    _pack_lhs_qai8dxp_asym_panel4_kernel,
    _pack_lhs_qsi8d32p_panel4_scalar_kernel,
    _pack_lhs_qsi8d32p_row_kernel,
    _q4_fused_decode_asym_sdot_kai_kernel,
    _q4_fused_decode_asym_g32_kai_sdot_kernel,
    _q4_fused_decode_asym_g128_sdot_kernel,
    _q4_fused_decode_asym_g128_kai_sdot_kernel,
    _q4_decode_asym_sdot_kai_kernel,
    _q4_decode_asym_g128_sdot_kernel,
    _q4_decode_sdot_kai_kernel,
    _q4_prefill_asym_i8mm_kai_kernel,
    _q4_prefill_asym_g128_i8mm_kernel,
    _q4_prefill_i8mm_kai_kernel,
)

logger = logging.getLogger(__name__)

BLOCK_LENGTH = 32
_USE_LEGACY_ROW_PACK = os.getenv(
    "FLAGGEMS_ARM_Q4_LEGACY_ROW_PACK", "0"
).lower() in {"1", "true", "on"}
_USE_FUSED_DECODE = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_DECODE", "1"
).lower() in {"1", "true", "on"}
_USE_FUSED_ASYM_DECODE = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_ASYM_DECODE", "1"
).lower() in {"1", "true", "on"}
_FUSED_G128_DECODE_OVERRIDE = os.getenv(
    "FLAGGEMS_ARM_Q4_FUSED_G128_DECODE"
)
_USE_COMPACT_G128_NORM = os.getenv(
    "FLAGGEMS_ARM_Q4_COMPACT_G128_NORM", "0"
).lower() in {"1", "true", "on"}
_USE_VLLM_FAST_APPLY = os.getenv(
    "FLAGGEMS_VLLM_FAST_APPLY",
    os.getenv("FLAGGEMS_Q4_FAST_APPLY", "0"),
).lower() in {"1", "true", "on"}
_STATS = {
    "prepared_linears": 0,
    "prepared_g128_linears": 0,
    "prepared_g32_linears": 0,
    "prepared_w8_linears": 0,
    "prepared_q4_lm_heads": 0,
    "prepared_w8_lm_heads": 0,
    "prepared_online_w8_linears": 0,
    "decode_codegen_calls": 0,
    "decode_programs": 0,
    "fused_decode_calls": 0,
    "fused_decode_programs": 0,
    "fused_rmsnorm_decode_calls": 0,
    "fused_rmsnorm_decode_programs": 0,
    "fused_rmsnorm_qk_norm_decode_calls": 0,
    "fused_rmsnorm_qk_norm_decode_programs": 0,
    "fused_add_rmsnorm_decode_calls": 0,
    "fused_add_rmsnorm_decode_programs": 0,
    "codegen_prefill_calls": 0,
    "m16_main_launches": 0,
    "m8_split_main_launches": 0,
    "panel4_pack_calls": 0,
    "panel4_tail_pack_calls": 0,
    "legacy_row_pack_calls": 0,
    "tail_m4_launches": 0,
    "tail_m8_launches": 0,
    "tail_m8_split_launches": 0,
    "tail_m12_launches": 0,
    "tail_m16_launches": 0,
}


def set_fused_decode_enabled(enabled: bool) -> bool:
    """Set the in-process Q4 decode A/B route and return its old value."""
    global _USE_FUSED_DECODE
    previous = _USE_FUSED_DECODE
    _USE_FUSED_DECODE = bool(enabled)
    return previous


def set_vllm_fast_apply_enabled(enabled: bool) -> bool:
    """Toggle cached prepared-layer quantized calls for same-engine A/B."""
    global _USE_VLLM_FAST_APPLY
    previous = _USE_VLLM_FAST_APPLY
    _USE_VLLM_FAST_APPLY = bool(enabled)
    return previous


def quantize_q4_0(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N,K]`` to signed Q4_0 values and FP16 K32 scales."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Q4 weight must be a floating-point [N,K] tensor")
    if weight.device.type != "cpu":
        raise ValueError(
            "ARM Q4 weight preparation supports CPU tensors only"
        )
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError(f"Q4 codegen requires N%4=0 and K%32=0; got {(n, k)}")
    if not torch.isfinite(weight).all():
        raise ValueError("Q4 weight must contain only finite values")
    blocks = weight.detach().float().reshape(
        n, k // BLOCK_LENGTH, BLOCK_LENGTH
    )
    max_indices = blocks.abs().argmax(dim=-1, keepdim=True)
    signed_max = torch.gather(blocks, -1, max_indices)
    scale = signed_max / -8.0
    reciprocal = torch.where(
        scale != 0, 1.0 / scale, torch.zeros_like(scale)
    )
    quantized = (blocks * reciprocal).round().clamp_(-8, 7).to(torch.int8)
    return quantized.reshape(n, k), scale.squeeze(-1).to(torch.float16)


def pack_rhs_qsi4c32p(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Build KAI's 72-byte ``[N4,K32]`` RHS block without a runtime call."""
    if quantized.dtype != torch.int8 or quantized.ndim != 2:
        raise ValueError("quantized weight must be an INT8 [N,K] tensor")
    if quantized.device.type != "cpu" or scale.device != quantized.device:
        raise ValueError(
            "Q4 values and scales must be CPU tensors on one device"
        )
    if not scale.is_floating_point():
        raise ValueError("Q4 scales must be floating point")
    n, k = quantized.shape
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("invalid Q4 tensor or scale shape")
    groups = k // BLOCK_LENGTH
    if scale.shape != (n, groups):
        raise ValueError("invalid Q4 tensor or scale shape")
    if not torch.isfinite(scale).all():
        raise ValueError("Q4 scales must contain only finite values")
    if quantized.min() < -8 or quantized.max() > 7:
        raise ValueError("Q4 values must be in the signed nibble range [-8,7]")

    rhs = torch.empty(
        (n // 4, groups, 72), dtype=torch.uint8, device=quantized.device
    )
    rhs[:, :, :8].view(torch.float16).copy_(
        scale.reshape(n // 4, 4, groups).permute(0, 2, 1).contiguous()
    )
    grouped = quantized.reshape(n, groups, 32)
    low = grouped[:, :, :16].reshape(n, groups, 2, 8).to(torch.int16) & 15
    high = grouped[:, :, 16:].reshape(n, groups, 2, 8).to(torch.int16) & 15
    data = (low | (high << 4)).to(torch.uint8)
    rhs[:, :, 8:].copy_(
        data.reshape(n // 4, 4, groups, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(n // 4, groups, 64)
    )
    return rhs.reshape(-1).contiguous()


def pack_rhs_qsi4c32p_asym(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Pack signed Q4, BF16 scales and scaled-int4 sums into an 80-byte ABI."""
    if quantized.dtype != torch.int8 or quantized.ndim != 2:
        raise ValueError("quantized weight must be an INT8 [N,K] tensor")
    n, k = quantized.shape
    groups = k // BLOCK_LENGTH
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("invalid asymmetric Q4 tensor shape")
    if scale.shape != (n, groups) or not scale.is_floating_point():
        raise ValueError("invalid asymmetric Q4 scale shape or dtype")
    if quantized.device.type != "cpu" or scale.device != quantized.device:
        raise ValueError("asymmetric Q4 packing currently supports CPU tensors")
    if not torch.isfinite(scale).all():
        raise ValueError("asymmetric Q4 scales must be finite")
    if int(quantized.min()) < -8 or int(quantized.max()) > 7:
        raise ValueError("Q4 values must be in [-8,7]")

    rhs = torch.empty(
        (n // 4, groups, 80), dtype=torch.uint8, device=quantized.device
    )
    rhs[:, :, :8].view(torch.bfloat16).copy_(
        scale.to(torch.bfloat16)
        .reshape(n // 4, 4, groups)
        .permute(0, 2, 1)
        .contiguous()
    )
    grouped = quantized.reshape(n, groups, 32)
    low = grouped[:, :, :16].reshape(n, groups, 2, 8).to(torch.int16) & 15
    high = grouped[:, :, 16:].reshape(n, groups, 2, 8).to(torch.int16) & 15
    data = (low | (high << 4)).to(torch.uint8)
    rhs[:, :, 8:72].copy_(
        data.reshape(n // 4, 4, groups, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(n // 4, groups, 64)
    )
    # The dot path sign-extends each nibble after multiplying it by 16.  Store
    # sums in the same scaled domain so zp correction is one vector multiply.
    scaled_sums = grouped.to(torch.int16).sum(dim=-1) * 16
    rhs[:, :, 72:80].view(torch.int16).copy_(
        scaled_sums.reshape(n // 4, 4, groups)
        .permute(0, 2, 1)
        .contiguous()
    )
    return rhs.reshape(-1).contiguous()


def pack_rhs_qsi4c32p_asym_compact(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Pack G32 Q4 with one scale-weighted correction per N4 tile.

    Each K32 record keeps four BF16 scales and 64 bytes of signed nibbles.
    The zero-point correction is linear across K, so a single FP32 vector at
    the end of the output tile replaces the INT16 vector formerly repeated in
    every 80-byte K32 record.
    """
    if quantized.dtype != torch.int8 or quantized.ndim != 2:
        raise ValueError("quantized weight must be an INT8 [N,K] tensor")
    n, k = quantized.shape
    groups = k // BLOCK_LENGTH
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("invalid compact asymmetric Q4 tensor shape")
    if scale.shape != (n, groups) or not scale.is_floating_point():
        raise ValueError("invalid compact asymmetric Q4 scale shape or dtype")
    if quantized.device.type != "cpu" or scale.device != quantized.device:
        raise ValueError("compact asymmetric Q4 packing supports CPU tensors")
    if not torch.isfinite(scale).all():
        raise ValueError("compact asymmetric Q4 scales must be finite")
    if int(quantized.min()) < -8 or int(quantized.max()) > 7:
        raise ValueError("Q4 values must be in [-8,7]")

    tile_stride = groups * 72 + 16
    rhs = torch.empty(
        (n // 4, tile_stride), dtype=torch.uint8, device=quantized.device
    )
    rhs_groups = rhs[:, : groups * 72].reshape(n // 4, groups, 72)
    scale_bf16 = scale.to(torch.bfloat16)
    rhs_groups[:, :, :8].view(torch.bfloat16).copy_(
        scale_bf16
        .reshape(n // 4, 4, groups)
        .permute(0, 2, 1)
        .contiguous()
    )
    grouped = quantized.reshape(n, groups, 32)
    low = grouped[:, :, :16].reshape(n, groups, 2, 8).to(torch.int16) & 15
    high = grouped[:, :, 16:].reshape(n, groups, 2, 8).to(torch.int16) & 15
    data = (low | (high << 4)).to(torch.uint8)
    rhs_groups[:, :, 8:].copy_(
        data.reshape(n // 4, 4, groups, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(n // 4, groups, 64)
    )

    scaled_sums = grouped.to(torch.int16).sum(dim=-1) * 16
    weighted_sums = (
        scaled_sums.to(torch.float32) * scale_bf16.to(torch.float32)
    ).sum(dim=1)
    rhs[:, groups * 72 :].view(torch.float32).copy_(
        weighted_sums.reshape(n // 4, 4)
    )
    return rhs.reshape(-1).contiguous()


def pack_rhs_qsi4c128p_asym(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Pack G128 scales plus one scale-weighted correction per N4 tile.

    The previous grouped-checkpoint route expanded every G128 scale over four
    K32 records and stored four separate correction sums.  This 264-byte
    ``[N4,K128]`` record keeps the same four compiler-visible K32 weight
    panels.  Zero-point correction is linear, so the scale-weighted Q4 sums
    are pre-accumulated once per output channel instead of stored per group.
    The resulting size matches the useful payload of KleidiAI's G128 pack.
    """
    if quantized.dtype != torch.int8 or quantized.ndim != 2:
        raise ValueError("quantized weight must be an INT8 [N,K] tensor")
    n, k = quantized.shape
    groups = k // 128
    if n <= 0 or k <= 0 or n % 4 or k % 128:
        raise ValueError("invalid asymmetric G128 Q4 tensor shape")
    if scale.shape != (n, groups) or not scale.is_floating_point():
        raise ValueError("invalid asymmetric G128 Q4 scale shape or dtype")
    if quantized.device.type != "cpu" or scale.device != quantized.device:
        raise ValueError("asymmetric G128 Q4 packing supports CPU tensors")
    if not torch.isfinite(scale).all():
        raise ValueError("asymmetric G128 Q4 scales must be finite")
    if int(quantized.min()) < -8 or int(quantized.max()) > 7:
        raise ValueError("Q4 values must be in [-8,7]")

    tile_stride = groups * 264 + 16
    rhs = torch.empty(
        (n // 4, tile_stride), dtype=torch.uint8, device=quantized.device
    )
    rhs_groups = rhs[:, : groups * 264].reshape(n // 4, groups, 264)
    scale_bf16 = scale.to(torch.bfloat16)
    rhs_groups[:, :, :8].view(torch.bfloat16).copy_(
        scale_bf16
        .reshape(n // 4, 4, groups)
        .permute(0, 2, 1)
        .contiguous()
    )
    groups32 = k // 32
    grouped32 = quantized.reshape(n, groups32, 32)
    low = grouped32[:, :, :16].reshape(n, groups32, 2, 8).to(torch.int16) & 15
    high = (
        grouped32[:, :, 16:].reshape(n, groups32, 2, 8).to(torch.int16) & 15
    )
    data32 = (low | (high << 4)).to(torch.uint8)
    packed32 = (
        data32.reshape(n // 4, 4, groups32, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(n // 4, groups32, 64)
    )
    rhs_groups[:, :, 8:264].copy_(
        packed32.reshape(n // 4, groups, 4, 64).reshape(
            n // 4, groups, 256
        )
    )
    # SDOT sees each signed nibble multiplied by sixteen.  Fold the per-group
    # sums with their BF16 scales now; runtime correction then needs one FP32
    # vector per output tile rather than an int16 vector in every K128 group.
    scaled_sums = quantized.reshape(n, groups, 128).to(torch.int16).sum(
        dim=-1
    ) * 16
    weighted_sums = (
        scaled_sums.to(torch.float32) * scale_bf16.to(torch.float32)
    ).sum(dim=1)
    rhs[:, groups * 264 :].view(torch.float32).copy_(
        weighted_sums.reshape(n // 4, 4)
    )
    return rhs.reshape(-1).contiguous()


def prepare_weight(
    weight: torch.Tensor,
    *,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Quantize and return only the packed production RHS blob.

    The KAI ABI is independently packed in groups of four output rows, so a
    large matrix can be prepared in row chunks without changing a byte of the
    final layout.  Keeping the temporary FP32 and INT8 tensors bounded is
    important for tied Qwen vocabulary matrices: materializing the complete
    ``[151936, 2048]`` weight in FP32 would otherwise require more than 1 GiB
    of transient memory in addition to the model and packed result.
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Q4 weight must be a floating-point [N,K] tensor")
    n, k = weight.shape
    if weight.device.type != "cpu":
        raise ValueError("ARM Q4 weight preparation supports CPU tensors only")
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError(f"Q4 codegen requires N%4=0 and K%32=0; got {(n, k)}")
    if chunk_rows <= 0 or chunk_rows % 4:
        raise ValueError("Q4 chunk_rows must be a positive multiple of four")
    if n <= chunk_rows:
        quantized, scale = quantize_q4_0(weight)
        return pack_rhs_qsi4c32p(quantized, scale)

    groups = k // BLOCK_LENGTH
    packed = torch.empty(
        (n // 4) * groups * 72,
        dtype=torch.uint8,
        device=weight.device,
    )
    byte_offset = 0
    for row_begin in range(0, n, chunk_rows):
        row_end = min(row_begin + chunk_rows, n)
        # N is required to be a multiple of four, and chunk_rows preserves
        # that boundary, including the final chunk.
        quantized, scale = quantize_q4_0(weight[row_begin:row_end])
        chunk = pack_rhs_qsi4c32p(quantized, scale)
        packed[byte_offset : byte_offset + chunk.numel()].copy_(chunk)
        byte_offset += chunk.numel()
    if byte_offset != packed.numel():
        raise RuntimeError("internal Q4 chunked-packing size mismatch")
    return packed


def prepare_weight_asym(
    weight: torch.Tensor,
    *,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Prepare G32 signed-Q4 weights for dynamic asymmetric A8 compute.

    Weight quantization deliberately matches the established HeadG32/Q4_0
    contract; the only difference from :func:`prepare_weight` is the metadata
    needed to correct a non-zero activation zero point.  Packing remains a
    model-load operation and matrix compute stays visible to Triton/LLVM.
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Q4 weight must be a floating-point [N,K] tensor")
    n, k = weight.shape
    if weight.device.type != "cpu":
        raise ValueError("ARM Q4 weight preparation supports CPU tensors only")
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError(f"Q4 codegen requires N%4=0 and K%32=0; got {(n, k)}")
    if chunk_rows <= 0 or chunk_rows % 4:
        raise ValueError("Q4 chunk_rows must be a positive multiple of four")

    groups = k // BLOCK_LENGTH
    packed = torch.empty(
        (n // 4) * groups * 80,
        dtype=torch.uint8,
        device=weight.device,
    )
    byte_offset = 0
    for row_begin in range(0, n, chunk_rows):
        row_end = min(row_begin + chunk_rows, n)
        quantized, scale = quantize_q4_0(weight[row_begin:row_end])
        chunk = pack_rhs_qsi4c32p_asym(
            quantized, scale.to(torch.bfloat16)
        )
        packed[byte_offset : byte_offset + chunk.numel()].copy_(chunk)
        byte_offset += chunk.numel()
    if byte_offset != packed.numel():
        raise RuntimeError("internal asymmetric Q4 packing size mismatch")
    return packed


def _tail_block(rows: int) -> int:
    if rows <= 0 or rows > 16:
        raise ValueError(f"tail row count must be in [1,16], got {rows}")
    return min(16, 4 * math.ceil(rows / 4))


def _decode_unroll(k: int) -> int:
    """Use four independent K32 bodies once K is latency dominated."""
    return 4 if k >= 4096 else 1


def _g128_decode_unroll(k: int) -> int:
    override = int(os.getenv("FLAGGEMS_ARM_Q4_G128_DECODE_UNROLL", "0"))
    if override not in (0, 1, 2, 4):
        raise ValueError("G128 decode unroll must be 0, 1, 2, or 4")
    return override or 1


def _use_fused_g128_decode(partitions: int) -> bool:
    """Fuse private activation packs only when partitions run concurrently."""
    if _FUSED_G128_DECODE_OVERRIDE is not None:
        value = _FUSED_G128_DECODE_OVERRIDE.lower()
        if value not in {"0", "1", "false", "true", "off", "on"}:
            raise ValueError(
                "FLAGGEMS_ARM_Q4_FUSED_G128_DECODE must be a boolean"
            )
        return value in {"1", "true", "on"}
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # Darwin has no sched_getaffinity.
        available_cpus = os.cpu_count() or 1
    return (
        partitions > 1
        and torch.get_num_threads() > 1
        and available_cpus >= partitions
    )


def _g128_prefill_block_m(rows: int) -> int:
    """Choose the measured G128 I8MM row tile for the current shape.

    M12/M16 keep twelve/sixteen FP32 result vectors live across the G128
    loop.  M16 amortizes launch overhead best for small aligned inputs, while
    M12's lower register pressure wins for long prefill.  For intermediate
    shapes, avoiding a separate tail launch is the strongest signal.
    """
    override = os.getenv("FLAGGEMS_ARM_Q4_G128_PREFILL_BLOCK_M")
    if override is not None:
        try:
            block_m = int(override)
        except ValueError as error:
            raise ValueError(
                "G128 prefill block M must be 4, 8, 12, or 16"
            ) from error
        if block_m not in (4, 8, 12, 16):
            raise ValueError(
                "G128 prefill block M must be 4, 8, 12, or 16"
            )
        return block_m

    if rows <= 16:
        return 16
    if rows >= 96 or rows % 12 == 0:
        return 12
    if rows % 16 == 0:
        return 16
    if rows % 8 == 0:
        return 8
    return 12


def _decode_partitions(k: int, n: int) -> int:
    """Choose independent output ranges for the CPU program grid."""
    override = int(os.getenv("FLAGGEMS_ARM_Q4_DECODE_PARTITIONS", "0"))
    if override < 0:
        raise ValueError("FLAGGEMS_ARM_Q4_DECODE_PARTITIONS must be >= 0")
    if override:
        return min(override, n // 4)
    threads = max(1, torch.get_num_threads())
    # Small projections do not amortize the OpenMP grid entry.  Every Qwen3
    # 1.7B projection is above this boundary; retain one program for focused
    # tiny-shape tests and low-end models.
    if k * n < 2 * 1024 * 1024:
        return 1
    return min(threads, n // 64)


def _decode_head_aligned_partitions(k: int, n: int, head_dim: int) -> int:
    """Retain parallelism while giving every program complete output heads."""
    candidate = _decode_partitions(k, n)
    for partitions in range(candidate, 0, -1):
        if n % partitions == 0 and (n // partitions) % head_dim == 0:
            return partitions
    return 1


def _contiguous_strides(shape) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * int(shape[index + 1])
    return tuple(strides)


def _workspace_bf16_view(
    typed_storage: torch.Tensor, byte_offset: int, shape
) -> torch.Tensor:
    """Create one final-shape view without slice/view/reshape chaining."""
    if byte_offset % torch.bfloat16.itemsize:
        raise ValueError("BF16 workspace offset is not aligned")
    return typed_storage.as_strided(
        tuple(int(item) for item in shape),
        _contiguous_strides(shape),
        byte_offset // torch.bfloat16.itemsize,
    )


def _decode_input(x: torch.Tensor, k: int) -> tuple[torch.Tensor, int]:
    """Avoid a reshape dispatch for already contiguous decode activations."""
    m = x.numel() // k
    if x.dtype == torch.bfloat16 and x.is_contiguous():
        return x, m
    return x.to(torch.bfloat16).reshape(m, k).contiguous(), m


def _use_m8_main_block() -> bool:
    """Use the legacy split-M8 schedule only when explicitly requested."""
    override = os.getenv("FLAGGEMS_ARM_Q4_M16_AS_M8")
    if override is not None:
        return override.lower() in {"1", "true", "on"}
    # ConvertDotToSVE2I8MM gives fixed-width M16 a distinct reverse-panel
    # schedule.  It removes LLVM's hot-loop register rotation and makes M16
    # faster than two M8 tiles on Neon as well as on SVE2-VL128.
    return False


def _run_codegen_prefill(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    shape = x.shape
    x_2d = x.to(torch.bfloat16).reshape(-1, k).contiguous()
    m = x_2d.shape[0]
    padded_m = 4 * math.ceil(m / 4)
    groups = k // BLOCK_LENGTH

    lhs_blob = torch.empty(
        (padded_m // 4) * groups * 136,
        dtype=torch.uint8,
        device=x.device,
    )
    if _USE_LEGACY_ROW_PACK:
        _pack_lhs_qsi8d32p_row_kernel[(padded_m,)](
            x_2d,
            lhs_blob.view(torch.float16),
            lhs_blob.view(torch.int8),
            m,
            x_2d.stride(0),
            K=k,
            num_warps=1,
            num_stages=1,
        )
        _STATS["legacy_row_pack_calls"] += 1
    else:
        full_panels = m // 4
        if full_panels:
            _pack_lhs_qsi8d32p_panel4_scalar_kernel[(full_panels,)](
                x_2d,
                lhs_blob.view(torch.float16),
                lhs_blob.view(torch.int8),
                m,
                x_2d.stride(0),
                K=k,
                FULL_PANEL=True,
                num_warps=1,
                num_stages=1,
            )
        tail_rows = m - full_panels * 4
        if tail_rows:
            lhs_byte_offset = full_panels * groups * 136
            _pack_lhs_qsi8d32p_panel4_scalar_kernel[(1,)](
                x_2d[full_panels * 4 :],
                lhs_blob[lhs_byte_offset:].view(torch.float16),
                lhs_blob[lhs_byte_offset:].view(torch.int8),
                tail_rows,
                x_2d.stride(0),
                K=k,
                FULL_PANEL=False,
                num_warps=1,
                num_stages=1,
            )
            _STATS["panel4_tail_pack_calls"] += 1
        _STATS["panel4_pack_calls"] += 1

    output = torch.empty((padded_m, n), dtype=torch.bfloat16, device=x.device)
    main_rows = (m // 16) * 16
    split_m16 = _use_m8_main_block()
    main_block_m = 8 if split_m16 else 16
    main_tiles = main_rows // main_block_m
    if main_tiles:
        _q4_prefill_i8mm_kai_kernel[(main_tiles, n // 4)](
            lhs_blob.view(torch.int8),
            lhs_blob.view(torch.float16),
            rhs.view(torch.uint8),
            rhs.view(torch.float16),
            output,
            N=n,
            K=k,
            BLOCK_M=main_block_m,
            num_warps=1,
            num_stages=1,
        )
        _STATS[
            "m8_split_main_launches" if split_m16 else "m16_main_launches"
        ] += 1

    remaining = m - main_rows
    if remaining:
        tail = _tail_block(remaining)
        split_tail_m16 = split_m16 and tail == 16
        tail_block_m = 8 if split_tail_m16 else tail
        tail_tiles = 2 if split_tail_m16 else 1
        lhs_byte_offset = (main_rows // 4) * groups * 136
        _q4_prefill_i8mm_kai_kernel[(tail_tiles, n // 4)](
            lhs_blob[lhs_byte_offset:].view(torch.int8),
            lhs_blob[lhs_byte_offset:].view(torch.float16),
            rhs.view(torch.uint8),
            rhs.view(torch.float16),
            output[main_rows:],
            N=n,
            K=k,
            BLOCK_M=tail_block_m,
            num_warps=1,
            num_stages=1,
        )
        if split_tail_m16:
            _STATS["tail_m8_split_launches"] += 1
        else:
            _STATS[f"tail_m{tail}_launches"] += 1

    _STATS["codegen_prefill_calls"] += 1
    return output[:m].reshape(*shape[:-1], n)


def _run_codegen_prefill_asym(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Token-asymmetric compressed-tensors prefill through ordinary Triton."""
    shape = x.shape
    x_2d = x.to(torch.bfloat16).reshape(-1, k).contiguous()
    m = x_2d.shape[0]
    padded_m = 4 * math.ceil(m / 4)
    lhs_panel_stride = 16 + 4 * k
    lhs_blob = torch.empty(
        (padded_m // 4) * lhs_panel_stride,
        dtype=torch.uint8,
        device=x.device,
    )
    _pack_lhs_qsi8d128p_asym_panel4_kernel[(padded_m // 4,)](
        x_2d,
        lhs_blob,
        m,
        x_2d.stride(0),
        K=k,
        num_warps=1,
        num_stages=1,
    )
    output = torch.empty((padded_m, n), dtype=torch.bfloat16, device=x.device)
    main_rows = (m // 16) * 16
    main_tiles = main_rows // 16
    if main_tiles:
        _q4_prefill_asym_i8mm_kai_kernel[(main_tiles, n // 4)](
            lhs_blob,
            rhs,
            output,
            N=n,
            K=k,
            BLOCK_M=16,
            num_warps=1,
            num_stages=1,
        )
    remaining = m - main_rows
    if remaining:
        tail = _tail_block(remaining)
        lhs_byte_offset = (main_rows // 4) * lhs_panel_stride
        _q4_prefill_asym_i8mm_kai_kernel[(1, n // 4)](
            lhs_blob[lhs_byte_offset:],
            rhs,
            output[main_rows:],
            N=n,
            K=k,
            BLOCK_M=tail,
            num_warps=1,
            num_stages=1,
        )
    return output[:m].reshape(*shape[:-1], n)


def _run_codegen_prefill_asym_kai(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """G32 prefill using the deployment ``qai8dxp_f32`` A8 contract."""
    shape = x.shape
    x_2d = x.to(torch.bfloat16).reshape(-1, k).contiguous()
    m = x_2d.shape[0]
    padded_m = 4 * math.ceil(m / 4)
    lhs_panel_stride = 32 + 4 * k
    lhs_blob = torch.empty(
        (padded_m // 4) * lhs_panel_stride,
        dtype=torch.uint8,
        device=x.device,
    )
    _pack_lhs_qai8dxp_asym_panel4_kernel[(padded_m // 4,)](
        x_2d,
        lhs_blob,
        m,
        x_2d.stride(0),
        K=k,
        num_warps=1,
        num_stages=1,
    )
    output = torch.empty((padded_m, n), dtype=torch.bfloat16, device=x.device)
    # A paired M8 grid is faster than the spilling M16 specialization on CIX,
    # while the joined M12 tail remains faster than M8+M4.  Start the M8 main
    # route only once there are at least sixteen rows.
    main_block = 8 if m >= 16 else 16
    main_rows = (m // main_block) * main_block
    if main_rows:
        _q4_prefill_asym_i8mm_kai_kernel[(main_rows // main_block, n // 4)](
            lhs_blob,
            rhs,
            output,
            N=n,
            K=k,
            BLOCK_M=main_block,
            LHS_KAI=True,
            num_warps=1,
            num_stages=1,
        )
    remaining = m - main_rows
    if remaining:
        tail = _tail_block(remaining)
        lhs_byte_offset = (main_rows // 4) * lhs_panel_stride
        _q4_prefill_asym_i8mm_kai_kernel[(1, n // 4)](
            lhs_blob[lhs_byte_offset:],
            rhs,
            output[main_rows:],
            N=n,
            K=k,
            BLOCK_M=tail,
            LHS_KAI=True,
            num_warps=1,
            num_stages=1,
        )
    return output[:m].reshape(*shape[:-1], n)


def _run_codegen_prefill_asym_g128(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """G128 grouped-checkpoint prefill through ordinary Triton I8MM."""
    shape = x.shape
    x_2d = x.to(torch.bfloat16).reshape(-1, k).contiguous()
    m = x_2d.shape[0]
    padded_m = 4 * math.ceil(m / 4)
    groups32 = k // BLOCK_LENGTH
    lhs_panel_stride = 16 + 4 * k
    lhs_blob = torch.empty(
        (padded_m // 4) * lhs_panel_stride,
        dtype=torch.uint8,
        device=x.device,
    )
    _pack_lhs_qsi8d128p_asym_panel4_kernel[(padded_m // 4,)](
        x_2d,
        lhs_blob,
        m,
        x_2d.stride(0),
        K=k,
        num_warps=1,
        num_stages=1,
    )
    output = torch.empty((padded_m, n), dtype=torch.bfloat16, device=x.device)
    main_block = _g128_prefill_block_m(m)
    main_rows = (m // main_block) * main_block
    main_tiles = main_rows // main_block
    if main_tiles:
        _q4_prefill_asym_g128_i8mm_kernel[(main_tiles, n // 4)](
            lhs_blob,
            rhs,
            output,
            N=n,
            K=k,
            BLOCK_M=main_block,
            num_warps=1,
            num_stages=1,
        )
    remaining = m - main_rows
    if remaining:
        tail = _tail_block(remaining)
        lhs_byte_offset = (main_rows // 4) * lhs_panel_stride
        _q4_prefill_asym_g128_i8mm_kernel[(1, n // 4)](
            lhs_blob[lhs_byte_offset:],
            rhs,
            output[main_rows:],
            N=n,
            K=k,
            BLOCK_M=tail,
            num_warps=1,
            num_stages=1,
        )
    return output[:m].reshape(*shape[:-1], n)


def _run_codegen_prefill_asym_g128_kai(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """G128 prefill with vLLM/KleidiAI FP32 activation quantization."""
    shape = x.shape
    x_2d = x.to(torch.bfloat16).reshape(-1, k).contiguous()
    m = x_2d.shape[0]
    padded_m = 4 * math.ceil(m / 4)
    lhs_panel_stride = 32 + 4 * k
    lhs_blob = torch.empty(
        (padded_m // 4) * lhs_panel_stride,
        dtype=torch.uint8,
        device=x.device,
    )
    _pack_lhs_qai8dxp_asym_panel4_kernel[(padded_m // 4,)](
        x_2d,
        lhs_blob,
        m,
        x_2d.stride(0),
        K=k,
        num_warps=1,
        num_stages=1,
    )
    output = torch.empty((padded_m, n), dtype=torch.bfloat16, device=x.device)
    main_block = _g128_prefill_block_m(m)
    main_rows = (m // main_block) * main_block
    main_tiles = main_rows // main_block
    if main_tiles:
        _q4_prefill_asym_g128_i8mm_kernel[(main_tiles, n // 4)](
            lhs_blob,
            rhs,
            output,
            N=n,
            K=k,
            BLOCK_M=main_block,
            LHS_KAI=True,
            num_warps=1,
            num_stages=1,
        )
    remaining = m - main_rows
    if remaining:
        tail = _tail_block(remaining)
        lhs_byte_offset = (main_rows // 4) * lhs_panel_stride
        _q4_prefill_asym_g128_i8mm_kernel[(1, n // 4)](
            lhs_blob[lhs_byte_offset:],
            rhs,
            output[main_rows:],
            N=n,
            K=k,
            BLOCK_M=tail,
            LHS_KAI=True,
            num_warps=1,
            num_stages=1,
        )
    return output[:m].reshape(*shape[:-1], n)


def _run_codegen_decode(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    groups = k // BLOCK_LENGTH
    partitions = _decode_partitions(k, n)
    if _USE_FUSED_DECODE:
        scratch_bytes = m * partitions * groups * 34
        output_bytes = m * n * torch.bfloat16.itemsize
        # One allocation owns private per-partition scratch followed by the
        # returned BF16 output.  The offset is always BF16-aligned because a
        # K32 decode group occupies 34 bytes.
        storage = torch.empty(
            (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
            dtype=torch.bfloat16,
            device=x.device,
        )
        output = _workspace_bf16_view(
            storage,
            scratch_bytes,
            (*shape[:-1], n),
        )
        _q4_fused_decode_sdot_kai_kernel[(m, partitions)](
            x_2d,
            storage,
            rhs,
            scratch_bytes,
            k,
            0,
            n // 4,
            K=k,
            N=n,
            UNROLL=_decode_unroll(k),
            num_warps=1,
            num_stages=1,
        )
        _STATS["fused_decode_calls"] += 1
        _STATS["fused_decode_programs"] += m * partitions
    else:
        output = torch.empty(
            (m, n), dtype=torch.bfloat16, device=x.device
        )
        lhs_blob = torch.empty(
            m * groups * 34, dtype=torch.uint8, device=x.device
        )
        _pack_lhs_qsi8d32p_decode_kernel[(m,)](
            x_2d,
            lhs_blob.view(torch.float16),
            lhs_blob.view(torch.int8),
            m,
            x_2d.stride(0),
            K=k,
            num_warps=1,
            num_stages=1,
        )
        _q4_decode_sdot_kai_kernel[(m, partitions)](
            lhs_blob,
            rhs,
            output,
            0,
            n // 4,
            K=k,
            N=n,
            UNROLL=_decode_unroll(k),
            num_warps=1,
            num_stages=1,
        )
    _STATS["decode_codegen_calls"] += 1
    _STATS["decode_programs"] += m * partitions
    if _USE_FUSED_DECODE:
        return output
    return output.reshape(*shape[:-1], n)


def _run_codegen_decode_asym(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Fused token-asymmetric activation pack and Q4 decode GEMV."""
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    groups = k // BLOCK_LENGTH
    partitions = _decode_partitions(k, n)
    scratch_copies = partitions if _USE_FUSED_ASYM_DECODE else 1
    scratch_bytes = m * scratch_copies * groups * 36
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(
        storage, scratch_bytes, (*shape[:-1], n)
    )
    if _USE_FUSED_ASYM_DECODE:
        _q4_fused_decode_asym_sdot_kai_kernel[(m, partitions)](
            x_2d,
            storage,
            rhs,
            scratch_bytes,
            k,
            0,
            n // 4,
            K=k,
            N=n,
            UNROLL=_decode_unroll(k),
            num_warps=1,
            num_stages=1,
        )
    else:
        _pack_lhs_qsi8d32p_asym_decode_kernel[(m,)](
            x_2d,
            storage,
            x_2d.stride(0),
            K=k,
            num_warps=1,
            num_stages=1,
        )
        _q4_decode_asym_sdot_kai_kernel[(m, partitions)](
            storage,
            rhs,
            output,
            0,
            n // 4,
            K=k,
            N=n,
            UNROLL=_decode_unroll(k),
            num_warps=1,
            num_stages=1,
        )
    return output


def _run_codegen_decode_asym_kai(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Fused ``qai8dxp_f32`` activation pack and G32 Q4 decode."""
    shape = x.shape
    x_2d, m = _decode_input(x.to(torch.bfloat16), k)
    partitions = _decode_partitions(k, n)
    lhs_row_bytes = 8 + k
    scratch_bytes = m * partitions * lhs_row_bytes
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(storage, scratch_bytes, (*shape[:-1], n))
    _q4_fused_decode_asym_g32_kai_sdot_kernel[(m, partitions)](
        x_2d,
        storage,
        rhs,
        scratch_bytes,
        x_2d.stride(0),
        0,
        n // 4,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        num_warps=1,
        num_stages=1,
    )
    return output


def _run_codegen_decode_asym_g128(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Pack token-asymmetric A8 once, then run partitioned G128 Q4 GEMV."""
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    partitions = _decode_partitions(k, n)
    use_fused = _use_fused_g128_decode(partitions)
    scratch_copies = partitions if use_fused else 1
    lhs_row_bytes = 4 + k
    scratch_bytes = m * scratch_copies * lhs_row_bytes
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(storage, scratch_bytes, (*shape[:-1], n))
    if use_fused:
        _q4_fused_decode_asym_g128_sdot_kernel[(m, partitions)](
            x_2d,
            storage,
            rhs,
            scratch_bytes,
            k,
            0,
            n // 4,
            K=k,
            N=n,
            UNROLL=_g128_decode_unroll(k),
            num_warps=1,
            num_stages=1,
        )
    else:
        _pack_lhs_qsi8d32p_asym_decode_kernel[(m,)](
            x_2d,
            storage,
            x_2d.stride(0),
            K=k,
            COMPACT=True,
            num_warps=1,
            num_stages=1,
        )
        _q4_decode_asym_g128_sdot_kernel[(m, partitions)](
            storage,
            rhs,
            output,
            output,
            0,
            n // 4,
            K=k,
            N=n,
            LHS_COMPACT=True,
            ADD_RESIDUAL=False,
            UNROLL=_g128_decode_unroll(k),
            num_warps=1,
            num_stages=1,
        )
    return output


def _run_codegen_decode_asym_g128_kai(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Fused vLLM/KleidiAI-compatible activation pack and Triton GEMV."""
    shape = x.shape
    x_2d, m = _decode_input(x.to(torch.bfloat16), k)
    partitions = _decode_partitions(k, n)
    lhs_row_bytes = 8 + k
    scratch_bytes = m * partitions * lhs_row_bytes
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(storage, scratch_bytes, (*shape[:-1], n))
    _q4_fused_decode_asym_g128_kai_sdot_kernel[(m, partitions)](
        x_2d,
        storage,
        rhs,
        scratch_bytes,
        x_2d.stride(0),
        0,
        n // 4,
        K=k,
        N=n,
        UNROLL=_g128_decode_unroll(k),
        num_warps=1,
        num_stages=1,
    )
    return output


def _run_codegen_decode_asym_g128_rmsnorm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    partitions = _decode_partitions(k, n)
    scratch_bytes = m * (
        (4 + k) if _USE_COMPACT_G128_NORM else (k // BLOCK_LENGTH) * 36
    )
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(storage, scratch_bytes, (*shape[:-1], n))
    _pack_lhs_qsi8d32p_asym_rmsnorm_decode_kernel[(m,)](
        x_2d,
        rms_weight,
        storage,
        x_2d.stride(0),
        rms_eps,
        K=k,
        NORM_TILE=16,
        COMPACT=_USE_COMPACT_G128_NORM,
        num_warps=1,
        num_stages=1,
    )
    _q4_decode_asym_g128_sdot_kernel[(m, partitions)](
        storage,
        rhs,
        output,
        output,
        0,
        n // 4,
        K=k,
        N=n,
        LHS_COMPACT=_USE_COMPACT_G128_NORM,
        ADD_RESIDUAL=False,
        UNROLL=_g128_decode_unroll(k),
        num_warps=1,
        num_stages=1,
    )
    return output


def _run_codegen_decode_asym_g128_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    residual_2d, residual_m = _decode_input(residual, k)
    if residual_m != m:
        raise ValueError("residual row count differs from input")
    partitions = _decode_partitions(k, n)
    scratch_bytes = m * (
        (4 + k) if _USE_COMPACT_G128_NORM else (k // BLOCK_LENGTH) * 36
    )
    residual_offset = scratch_bytes
    residual_bytes = m * k * torch.bfloat16.itemsize
    output_offset = residual_offset + residual_bytes
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (output_offset + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    updated_residual = _workspace_bf16_view(storage, residual_offset, shape)
    output = _workspace_bf16_view(
        storage, output_offset, (*shape[:-1], n)
    )
    _pack_lhs_qsi8d32p_asym_add_rmsnorm_decode_kernel[(m,)](
        x_2d,
        residual_2d,
        rms_weight,
        storage,
        updated_residual,
        x_2d.stride(0),
        rms_eps,
        K=k,
        NORM_TILE=16,
        COMPACT=_USE_COMPACT_G128_NORM,
        num_warps=1,
        num_stages=1,
    )
    _q4_decode_asym_g128_sdot_kernel[(m, partitions)](
        storage,
        rhs,
        output,
        output,
        0,
        n // 4,
        K=k,
        N=n,
        LHS_COMPACT=_USE_COMPACT_G128_NORM,
        ADD_RESIDUAL=False,
        UNROLL=_g128_decode_unroll(k),
        num_warps=1,
        num_stages=1,
    )
    return output, updated_residual


def _run_codegen_decode_asym_rmsnorm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    groups = k // BLOCK_LENGTH
    partitions = _decode_partitions(k, n)
    scratch_bytes = m * groups * 36
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(
        storage, scratch_bytes, (*shape[:-1], n)
    )
    _pack_lhs_qsi8d32p_asym_rmsnorm_decode_kernel[(m,)](
        x_2d,
        rms_weight,
        storage,
        x_2d.stride(0),
        rms_eps,
        K=k,
        NORM_TILE=16,
        num_warps=1,
        num_stages=1,
    )
    _q4_decode_asym_sdot_kai_kernel[(m, partitions)](
        storage,
        rhs,
        output,
        0,
        n // 4,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        num_warps=1,
        num_stages=1,
    )
    return output


def _run_codegen_decode_asym_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    residual_2d, residual_m = _decode_input(residual, k)
    if residual_m != m:
        raise ValueError("residual row count differs from input")
    groups = k // BLOCK_LENGTH
    partitions = _decode_partitions(k, n)
    scratch_bytes = m * groups * 36
    residual_offset = scratch_bytes
    residual_bytes = m * k * torch.bfloat16.itemsize
    output_offset = residual_offset + residual_bytes
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (output_offset + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    updated_residual = _workspace_bf16_view(
        storage, residual_offset, shape
    )
    output = _workspace_bf16_view(
        storage, output_offset, (*shape[:-1], n)
    )
    _pack_lhs_qsi8d32p_asym_add_rmsnorm_decode_kernel[(m,)](
        x_2d,
        residual_2d,
        rms_weight,
        storage,
        updated_residual,
        x_2d.stride(0),
        rms_eps,
        K=k,
        NORM_TILE=16,
        num_warps=1,
        num_stages=1,
    )
    _q4_decode_asym_sdot_kai_kernel[(m, partitions)](
        storage,
        rhs,
        output,
        0,
        n // 4,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        num_warps=1,
        num_stages=1,
    )
    return output, updated_residual


def _run_codegen_decode_rmsnorm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Decode-only ordinary-Triton RMSNorm + Q4 activation pack + GEMV."""
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    groups = k // BLOCK_LENGTH
    partitions = _decode_partitions(k, n)
    scratch_bytes = m * partitions * groups * 34
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(
        storage,
        scratch_bytes,
        (*shape[:-1], n),
    )
    _q4_fused_rmsnorm_decode_sdot_kai_kernel[(m, partitions)](
        x_2d,
        rms_weight,
        storage,
        rhs,
        scratch_bytes,
        k,
        0,
        n // 4,
        rms_eps,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        NORM_TILE=16,
        num_warps=1,
        num_stages=1,
    )
    _STATS["decode_codegen_calls"] += 1
    _STATS["decode_programs"] += m * partitions
    _STATS["fused_rmsnorm_decode_calls"] += 1
    _STATS["fused_rmsnorm_decode_programs"] += m * partitions
    return output


def _run_codegen_decode_rmsnorm_qk_norm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    qk_weight: torch.Tensor,
    qk_eps: float,
    q_elements: int,
    k_elements: int,
    head_dim: int,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Decode input norm, joined QKV GEMV, and owned Q/K head norms."""
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    groups = k // BLOCK_LENGTH
    partitions = _decode_head_aligned_partitions(k, n, head_dim)
    scratch_bytes = m * partitions * groups * 34
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (scratch_bytes + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    output = _workspace_bf16_view(
        storage,
        scratch_bytes,
        (*shape[:-1], n),
    )
    _q4_fused_rmsnorm_qk_norm_decode_sdot_kai_kernel[(m, partitions)](
        x_2d,
        rms_weight,
        qk_weight,
        storage,
        rhs,
        scratch_bytes,
        k,
        0,
        n // 4,
        rms_eps,
        qk_eps,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        Q_ELEMENTS=q_elements,
        K_ELEMENTS=k_elements,
        HEAD_DIM=head_dim,
        NORM_TILE=16,
        num_warps=1,
        num_stages=1,
    )
    _STATS["decode_codegen_calls"] += 1
    _STATS["decode_programs"] += m * partitions
    _STATS["fused_rmsnorm_decode_calls"] += 1
    _STATS["fused_rmsnorm_decode_programs"] += m * partitions
    _STATS["fused_rmsnorm_qk_norm_decode_calls"] += 1
    _STATS["fused_rmsnorm_qk_norm_decode_programs"] += m * partitions
    return output


def _run_codegen_decode_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode residual add + RMSNorm + activation pack + Q4 GEMV."""
    shape = x.shape
    x_2d, m = _decode_input(x, k)
    residual_2d, residual_m = _decode_input(residual, k)
    if residual_m != m:
        raise ValueError("residual row count differs from input")
    groups = k // BLOCK_LENGTH
    partitions = _decode_partitions(k, n)
    quant_bytes = m * partitions * groups * 34
    summed_byte_offset = quant_bytes
    summed_bytes = m * partitions * k * torch.bfloat16.itemsize
    output_byte_offset = summed_byte_offset + summed_bytes
    output_bytes = m * n * torch.bfloat16.itemsize
    storage = torch.empty(
        (output_byte_offset + output_bytes) // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=x.device,
    )
    updated_residual = _workspace_bf16_view(
        storage, summed_byte_offset, shape
    )
    output = _workspace_bf16_view(
        storage, output_byte_offset, (*shape[:-1], n)
    )
    _q4_fused_add_rmsnorm_decode_sdot_kai_kernel[(m, partitions)](
        x_2d,
        residual_2d,
        rms_weight,
        storage,
        rhs,
        summed_byte_offset,
        output_byte_offset,
        k,
        0,
        n // 4,
        rms_eps,
        K=k,
        N=n,
        UNROLL=_decode_unroll(k),
        NORM_TILE=16,
        num_warps=1,
        num_stages=1,
    )
    _STATS["decode_codegen_calls"] += 1
    _STATS["decode_programs"] += m * partitions
    _STATS["fused_add_rmsnorm_decode_calls"] += 1
    _STATS["fused_add_rmsnorm_decode_programs"] += m * partitions
    return (
        output,
        updated_residual,
    )


def linear_w4a8(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Runtime router: Triton SDOT decode, otherwise target-sized prefill."""
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError(
            "Q4 dimensions require N>0, K>0, N%4=0 and K%32=0"
        )
    if x.device.type != "cpu" or rhs.device.type != "cpu":
        raise ValueError("ARM Q4 codegen currently supports CPU tensors only")
    if (
        x.ndim == 0
        or x.numel() == 0
        or not x.is_floating_point()
        or x.shape[-1] != k
        or rhs.device != x.device
    ):
        raise ValueError("Q4 input/RHS shape or device mismatch")
    if rhs.dtype != torch.uint8 or not rhs.is_contiguous():
        raise ValueError("KAI Q4 RHS must be a contiguous UINT8 blob")
    expected_rhs_bytes = (n // 4) * (k // BLOCK_LENGTH) * 72
    if rhs.numel() != expected_rhs_bytes:
        raise ValueError(
            "invalid KAI Q4 RHS: requires N%4=0, K%32=0 and "
            f"{expected_rhs_bytes} packed bytes"
        )
    m = x.numel() // k
    if m < 4:
        return _run_codegen_decode(x, rhs, n, k)
    return _run_codegen_prefill(x, rhs, n, k)


def linear_w4a8_asym(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Compressed-tensors token-asymmetric A8 x grouped signed-Q4 router."""
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("asymmetric Q4 requires N%4=0 and K%32=0")
    if (
        x.device.type != "cpu"
        or x.ndim == 0
        or x.numel() == 0
        or not x.is_floating_point()
        or x.shape[-1] != k
        or rhs.device != x.device
        or rhs.dtype != torch.uint8
        or not rhs.is_contiguous()
    ):
        raise ValueError("asymmetric Q4 input/RHS contract mismatch")
    expected_rhs_bytes = (n // 4) * (k // BLOCK_LENGTH) * 80
    if rhs.numel() != expected_rhs_bytes:
        raise ValueError("invalid asymmetric Q4 RHS byte count")
    m = x.numel() // k
    if m < 4:
        return _run_codegen_decode_asym(x, rhs, n, k)
    return _run_codegen_prefill_asym(x, rhs, n, k)


def linear_w4a8_asym_kai(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """G32 Q4 router with KleidiAI ``qai8dxp_f32`` activation semantics."""
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("KAI asymmetric Q4 requires N%4=0 and K%32=0")
    expected_rhs_bytes = (n // 4) * (k // BLOCK_LENGTH) * 80
    if (
        x.device.type != "cpu"
        or x.ndim == 0
        or x.numel() == 0
        or not x.is_floating_point()
        or x.shape[-1] != k
        or rhs.device != x.device
        or rhs.dtype != torch.uint8
        or not rhs.is_contiguous()
        or rhs.numel() != expected_rhs_bytes
    ):
        raise ValueError("KleidiAI-compatible G32 Q4 contract mismatch")
    m = x.numel() // k
    if m < 4:
        return _run_codegen_decode_asym_kai(x, rhs, n, k)
    return _run_codegen_prefill_asym_kai(x, rhs, n, k)


def linear_w4a8_asym_g128(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Token-asymmetric A8 x signed-Q4 with native G128 RHS metadata."""
    if n <= 0 or k <= 0 or n % 4 or k % 128:
        raise ValueError("asymmetric G128 Q4 requires N%4=0 and K%128=0")
    if (
        x.device.type != "cpu"
        or x.ndim == 0
        or x.numel() == 0
        or not x.is_floating_point()
        or x.shape[-1] != k
        or rhs.device != x.device
        or rhs.dtype != torch.uint8
        or not rhs.is_contiguous()
    ):
        raise ValueError("asymmetric G128 Q4 input/RHS contract mismatch")
    expected_rhs_bytes = (n // 4) * ((k // 128) * 264 + 16)
    if rhs.numel() != expected_rhs_bytes:
        raise ValueError("invalid asymmetric G128 Q4 RHS byte count")
    m = x.numel() // k
    if m < 4:
        return _run_codegen_decode_asym_g128(x, rhs, n, k)
    return _run_codegen_prefill_asym_g128(x, rhs, n, k)


def linear_w4a8_asym_g128_kai(
    x: torch.Tensor,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """G128 router matching vLLM/KleidiAI FP32 activation quantization."""
    if n <= 0 or k <= 0 or n % 4 or k % 128:
        raise ValueError("asymmetric G128 Q4 requires N%4=0 and K%128=0")
    expected_rhs_bytes = (n // 4) * ((k // 128) * 264 + 16)
    if (
        x.device.type != "cpu"
        or x.ndim == 0
        or x.numel() == 0
        or not x.is_floating_point()
        or x.shape[-1] != k
        or rhs.device != x.device
        or rhs.dtype != torch.uint8
        or not rhs.is_contiguous()
        or rhs.numel() != expected_rhs_bytes
    ):
        raise ValueError("KleidiAI-compatible G128 Q4 contract mismatch")
    m = x.numel() // k
    if m < 4:
        return _run_codegen_decode_asym_g128_kai(x, rhs, n, k)
    return _run_codegen_prefill_asym_g128_kai(x, rhs, n, k)


def linear_w4a8_asym_g128_rmsnorm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Decode RMSNorm plus native-G128 asymmetric Q4 projection."""
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.shape[-1] != k
        or x.numel() // k >= 4
        or rms_weight.dtype != torch.bfloat16
        or rms_weight.shape != (k,)
        or not rms_weight.is_contiguous()
        or rhs.dtype != torch.uint8
        or rhs.numel() != (n // 4) * ((k // 128) * 264 + 16)
    ):
        raise ValueError("invalid asymmetric G128 Q4 RMSNorm decode contract")
    return _run_codegen_decode_asym_g128_rmsnorm(
        x, rms_weight, float(rms_eps), rhs, n, k
    )


def linear_w4a8_asym_g128_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode add/RMSNorm plus native-G128 asymmetric Q4 projection."""
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.shape[-1] != k
        or x.numel() // k >= 4
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or rms_weight.dtype != torch.bfloat16
        or rms_weight.shape != (k,)
        or not rms_weight.is_contiguous()
        or rhs.dtype != torch.uint8
        or rhs.numel() != (n // 4) * ((k // 128) * 264 + 16)
    ):
        raise ValueError(
            "invalid asymmetric G128 Q4 add/RMSNorm decode contract"
        )
    return _run_codegen_decode_asym_g128_add_rmsnorm(
        x, residual, rms_weight, float(rms_eps), rhs, n, k
    )


def linear_w4a8_asym_rmsnorm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Decode RMSNorm + compressed-tensors asymmetric Q4 projection."""
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.shape[-1] != k
        or x.numel() // k >= 4
        or rms_weight.dtype != torch.bfloat16
        or rms_weight.shape != (k,)
        or not rms_weight.is_contiguous()
        or rhs.dtype != torch.uint8
        or rhs.numel() != (n // 4) * (k // BLOCK_LENGTH) * 80
    ):
        raise ValueError("invalid asymmetric Q4 RMSNorm decode contract")
    return _run_codegen_decode_asym_rmsnorm(
        x, rms_weight, float(rms_eps), rhs, n, k
    )


def linear_w4a8_asym_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode add/RMSNorm + compressed-tensors asymmetric Q4 projection."""
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.shape[-1] != k
        or x.numel() // k >= 4
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or rms_weight.dtype != torch.bfloat16
        or rms_weight.shape != (k,)
        or not rms_weight.is_contiguous()
        or rhs.dtype != torch.uint8
        or rhs.numel() != (n // 4) * (k // BLOCK_LENGTH) * 80
    ):
        raise ValueError("invalid asymmetric Q4 add/RMSNorm decode contract")
    return _run_codegen_decode_asym_add_rmsnorm(
        x, residual, rms_weight, float(rms_eps), rhs, n, k
    )


def linear_w4a8_rmsnorm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Fuse Qwen-compatible BF16 RMSNorm into an M<4 Q4 projection."""
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.ndim == 0
        or x.numel() == 0
        or x.shape[-1] != k
        or x.numel() // k >= 4
    ):
        raise ValueError("fused RMSNorm Q4 requires CPU BF16 M=1..3")
    if (
        rms_weight.device != x.device
        or rms_weight.dtype != torch.bfloat16
        or rms_weight.shape != (k,)
        or not rms_weight.is_contiguous()
    ):
        raise ValueError("fused RMSNorm Q4 requires contiguous BF16 [K] weight")
    if not math.isfinite(float(rms_eps)) or float(rms_eps) < 0.0:
        raise ValueError("fused RMSNorm Q4 epsilon must be finite and nonnegative")
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("fused RMSNorm Q4 requires N%4=0 and K%32=0")
    if rhs.dtype != torch.uint8 or not rhs.is_contiguous():
        raise ValueError("fused RMSNorm Q4 requires contiguous KAI UINT8 RHS")
    expected_rhs_bytes = (n // 4) * (k // BLOCK_LENGTH) * 72
    if rhs.device != x.device or rhs.numel() != expected_rhs_bytes:
        raise ValueError("invalid fused RMSNorm Q4 RHS shape or device")
    return _run_codegen_decode_rmsnorm(
        x, rms_weight, float(rms_eps), rhs, n, k
    )


def linear_w4a8_rmsnorm_qk_norm(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    qk_weight: torch.Tensor,
    qk_eps: float,
    q_elements: int,
    k_elements: int,
    head_dim: int,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Fuse input and Q/K head RMSNorm into a joined decode projection."""
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.ndim == 0
        or x.numel() == 0
        or x.shape[-1] != k
        or x.numel() // k >= 4
    ):
        raise ValueError("fused Q/K-norm Q4 requires CPU BF16 M=1..3")
    if (
        rms_weight.device != x.device
        or rms_weight.dtype != torch.bfloat16
        or rms_weight.shape != (k,)
        or not rms_weight.is_contiguous()
    ):
        raise ValueError("invalid input RMSNorm weight")
    if (
        head_dim <= 0
        or head_dim % 16
        or q_elements <= 0
        or k_elements <= 0
        or q_elements % head_dim
        or k_elements % head_dim
        or q_elements + k_elements > n
    ):
        raise ValueError("Q/K output ranges must contain whole aligned heads")
    if (
        qk_weight.device != x.device
        or qk_weight.dtype != torch.bfloat16
        or qk_weight.shape != (2 * head_dim,)
        or not qk_weight.is_contiguous()
    ):
        raise ValueError("Q/K RMSNorm weight must be contiguous BF16 [2*D]")
    if not all(
        math.isfinite(float(eps)) and float(eps) >= 0.0
        for eps in (rms_eps, qk_eps)
    ):
        raise ValueError("RMSNorm epsilons must be finite and nonnegative")
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("fused Q/K-norm Q4 requires N%4=0 and K%32=0")
    expected_rhs_bytes = (n // 4) * (k // BLOCK_LENGTH) * 72
    if (
        rhs.device != x.device
        or rhs.dtype != torch.uint8
        or not rhs.is_contiguous()
        or rhs.numel() != expected_rhs_bytes
    ):
        raise ValueError("invalid fused Q/K-norm Q4 RHS")
    return _run_codegen_decode_rmsnorm_qk_norm(
        x,
        rms_weight,
        float(rms_eps),
        qk_weight,
        float(qk_eps),
        q_elements,
        k_elements,
        head_dim,
        rhs,
        n,
        k,
    )


def linear_w4a8_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    rhs: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse residual add and Qwen BF16 RMSNorm into an M<4 Q4 projection."""
    if (
        x.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or x.ndim == 0
        or x.numel() == 0
        or x.shape[-1] != k
        or x.numel() // k >= 4
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("fused add-RMSNorm Q4 requires matching CPU BF16 M=1..3")
    if (
        rms_weight.device != x.device
        or rms_weight.dtype != torch.bfloat16
        or rms_weight.shape != (k,)
        or not rms_weight.is_contiguous()
    ):
        raise ValueError(
            "fused add-RMSNorm Q4 requires contiguous BF16 [K] weight"
        )
    if not math.isfinite(float(rms_eps)) or float(rms_eps) < 0.0:
        raise ValueError("fused add-RMSNorm Q4 epsilon must be nonnegative")
    if n <= 0 or k <= 0 or n % 4 or k % BLOCK_LENGTH:
        raise ValueError("fused add-RMSNorm Q4 requires N%4=0 and K%32=0")
    expected_rhs_bytes = (n // 4) * (k // BLOCK_LENGTH) * 72
    if (
        rhs.device != x.device
        or rhs.dtype != torch.uint8
        or not rhs.is_contiguous()
        or rhs.numel() != expected_rhs_bytes
    ):
        raise ValueError("invalid fused add-RMSNorm Q4 RHS")
    return _run_codegen_decode_add_rmsnorm(
        x, residual, rms_weight, float(rms_eps), rhs, n, k
    )


@torch.library.custom_op("flag_gems::arm_q4_linear", mutates_args=())
def _linear_w4a8_router_op(
    x: torch.Tensor,
    rhs_codegen: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Opaque runtime shape router for vLLM's trace-once compiled graph."""
    return linear_w4a8(x, rhs_codegen, n, k)


@_linear_w4a8_router_op.register_fake
def _(
    x: torch.Tensor,
    rhs_codegen: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n), dtype=torch.bfloat16)


def _make_vllm_linear(
    rhs_codegen: torch.Tensor,
    n: int,
    k: int,
    runtime: str = "python",
    asymmetric: bool = False,
):
    """Create the callable stored in vLLM's ``layer.cpu_linear`` slot."""
    def cpu_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        if runtime == "libtriton_jit":
            op = (
                torch.ops.triton_jit_cpu.q4_linear_g32_asym
                if asymmetric
                else torch.ops.triton_jit_cpu.q4_linear
            )
            out = op(x, rhs_codegen, n, k)
        # Keep an opaque operator while Dynamo captures the model graph, but
        # do not pay the Python -> dispatcher -> Python custom-op round trip
        # in eager vLLM. The direct route invokes the exact same production
        # Triton kernels and preserves runtime-M routing.
        elif torch.compiler.is_compiling():
            out = torch.ops.flag_gems.arm_q4_linear(
                x, rhs_codegen, n, k
            )
        else:
            out = linear_w4a8(x, rhs_codegen, n, k)
        return out + bias.to(out.dtype) if bias is not None else out

    return cpu_linear


def prepare_w8_weight(
    weight: torch.Tensor,
    *,
    chunk_rows: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP32 ``[N,K]`` weight to legacy symmetric-A8 packs.

    The weight contract matches the compressed-tensors MiniCPM W8 checkpoint:
    symmetric signed INT8 with one FP32 scale per output channel.  Rows are
    quantized in bounded chunks because a vocabulary projection is hundreds
    of MiB even before an eager FP32 conversion.  Both returned layouts are
    ordinary-Triton inputs: block-major SDOT for decode and N4/K8 I8MM for
    prefill.  The deployment-guide route uses :func:`prepare_w8_weight_kai`;
    this function is retained as a controlled symmetric-A8 regression oracle.
    """
    from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
        pack_weights_i8mm_kai,
        pack_weights_sdot,
        pack_weights_sdot_blocked,
        select_w8_decode_tile_n,
    )

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("W8 weight must be a floating-point [N,K] tensor")
    if weight.device.type != "cpu":
        raise ValueError("ARM W8 weight preparation supports CPU tensors only")
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % 64 or k % 32:
        raise ValueError(f"W8 codegen requires N%64=0 and K%32=0; got {(n, k)}")
    if chunk_rows <= 0 or chunk_rows % 64:
        raise ValueError("W8 chunk_rows must be a positive multiple of 64")

    panel_stride = 4 * k + 16
    decode_block_n = select_w8_decode_tile_n(n, 64)
    decode = torch.empty(
        (n // decode_block_n, k // 4, decode_block_n // 4, 4, 4),
        dtype=torch.int8,
        device=weight.device,
    )
    prefill = torch.empty(
        (n // 4, panel_stride),
        dtype=torch.int8,
        device=weight.device,
    )
    scales = torch.empty((n,), dtype=torch.float32, device=weight.device)

    for row_begin in range(0, n, chunk_rows):
        row_end = min(row_begin + chunk_rows, n)
        values = weight[row_begin:row_end].detach().to(torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("W8 weight must contain only finite values")
        scale = (values.abs().amax(dim=1) / 127.0).clamp_min_(1.0e-8)
        quantized = (
            (values / scale[:, None])
            .round()
            .clamp_(-127, 127)
            .to(torch.int8)
        )

        decode_chunk = pack_weights_sdot_blocked(
            pack_weights_sdot(quantized.T.contiguous()), decode_block_n
        )
        decode[
            row_begin // decode_block_n : row_end // decode_block_n
        ].copy_(decode_chunk)
        prefill_chunk = pack_weights_i8mm_kai(quantized, scale).reshape(
            (row_end - row_begin) // 4, panel_stride
        )
        prefill[row_begin // 4 : row_end // 4].copy_(prefill_chunk)
        scales[row_begin:row_end].copy_(scale)

    return (
        decode.reshape(-1).contiguous(),
        prefill.reshape(-1).contiguous(),
        scales.contiguous(),
    )


def pack_rhs_w8_symmetric(
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Pack symmetric per-channel W8 into a compact N4/K8 Triton ABI."""
    if quantized.dtype != torch.int8 or quantized.ndim != 2:
        raise ValueError("compact W8 values must be an INT8 [N,K] tensor")
    if quantized.device.type != "cpu" or scales.device != quantized.device:
        raise ValueError("compact W8 packing requires CPU tensors on one device")
    n, k = quantized.shape
    if n <= 0 or k <= 0 or n % 4 or k % 32:
        raise ValueError("compact W8 packing requires N%4=0 and K%32=0")
    scales = scales.reshape(-1).to(torch.float32)
    if scales.numel() != n or not torch.isfinite(scales).all():
        raise ValueError(
            "compact W8 packing requires one finite FP32 scale per output"
        )

    panel_stride = 4 * k + 16
    packed = torch.empty(
        (n // 4, panel_stride), dtype=torch.uint8, device=quantized.device
    )
    interleaved = (
        quantized.reshape(n // 4, 4, k // 8, 8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(n // 4, 4 * k)
    )
    packed[:, : 4 * k].view(torch.int8).copy_(interleaved)
    packed[:, 4 * k :].view(torch.float32).copy_(
        scales.reshape(n // 4, 4)
    )
    return packed.view(torch.int8).reshape(-1).contiguous()


def pack_rhs_qsi8cxp(
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Compatibility alias for the compact symmetric W8 packer.

    The old implementation stored asymmetric correction metadata that was
    incompatible with symmetric-input checkpoints.  Keep the exported name
    so downstream callers do not break while routing it to the corrected ABI.
    """
    return pack_rhs_w8_symmetric(quantized, scales)


def prepare_w8_weight_kai(
    weight: torch.Tensor,
    *,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Quantize floating-point W8 into the compact symmetric N4/K8 ABI."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("W8 weight must be a floating-point [N,K] tensor")
    if weight.device.type != "cpu":
        raise ValueError("ARM W8 weight preparation supports CPU tensors only")
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % 4 or k % 32:
        raise ValueError(
            f"compact W8 requires N%4=0 and K%32=0; got {(n, k)}"
        )
    if chunk_rows <= 0 or chunk_rows % 4:
        raise ValueError("W8 chunk_rows must be a positive multiple of four")

    panel_stride = 4 * k + 16
    packed = torch.empty(
        (n // 4, panel_stride), dtype=torch.int8, device=weight.device
    )
    for row_begin in range(0, n, chunk_rows):
        row_end = min(row_begin + chunk_rows, n)
        values = weight[row_begin:row_end].detach().to(torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("W8 weight must contain only finite values")
        scales = (values.abs().amax(dim=1) / 127.0).clamp_min_(1.0e-8)
        quantized = (
            (values / scales[:, None])
            .round()
            .clamp_(-127, 127)
            .to(torch.int8)
        )
        chunk = pack_rhs_w8_symmetric(quantized, scales).reshape(
            (row_end - row_begin) // 4, panel_stride
        )
        packed[row_begin // 4 : row_end // 4].copy_(chunk)
    return packed.reshape(-1).contiguous()


def _make_vllm_w8_linear(
    rhs: torch.Tensor,
    n: int,
    k: int,
):
    """Create a callable for symmetric dynamic-A8 x compact N4/K8 W8."""
    def cpu_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        source_dtype = x.dtype
        x_bf16 = x if source_dtype == torch.bfloat16 else x.to(torch.bfloat16)
        out = torch.ops.triton_jit_cpu.w8_linear_kai(
            x_bf16.contiguous(), rhs, n, k
        )
        if source_dtype != torch.bfloat16:
            out = out.to(source_dtype)
        return out + bias.to(out.dtype) if bias is not None else out

    return cpu_linear


def _vllm_dynamic4bit_kernel(layer: torch.nn.Module):
    """Resolve the MP kernel nested under vLLM's quantization scheme."""
    scheme = getattr(layer, "scheme", None)
    return getattr(scheme, "kernel", None)


def prepare_vllm_q4_g32_pair(
    first_layer: torch.nn.Module,
    second_layer: torch.nn.Module,
) -> object | None:
    """Resolve two G32 projections once and return their decode hot path.

    This is intentionally a narrow helper for the Qwen GDN qkvz/ba pair.  It
    resolves vLLM's nested quantization objects and hybrid RHS slices once;
    repeating that Python introspection in 48 layers per token erases the
    small native-kernel saving.  The returned callable still rejects prefill.
    """
    if (
        getattr(first_layer, "bias", None) is not None
        or getattr(second_layer, "bias", None) is not None
        or getattr(first_layer, "tp_size", 1) != 1
        or getattr(second_layer, "tp_size", 1) != 1
    ):
        return None
    first_kernel = _vllm_dynamic4bit_kernel(first_layer)
    second_kernel = _vllm_dynamic4bit_kernel(second_layer)
    if first_kernel is None or second_kernel is None:
        return None
    first_shape = getattr(
        first_kernel, "_flag_gems_libtriton_jit_q4_shape", None
    )
    second_shape = getattr(
        second_kernel, "_flag_gems_libtriton_jit_q4_shape", None
    )
    if first_shape is None or second_shape is None:
        return None
    n0, k0, group0 = first_shape
    n1, k1, group1 = second_shape
    if group0 != group1 or group0 not in {32, 128} or k0 != k1:
        return None

    if group0 == 32 and not hasattr(
        torch.ops.triton_jit_cpu, "q4_linear_g32_asym_compact_pair"
    ):
        return None
    if group0 == 128 and not hasattr(
        torch.ops.triton_jit_cpu, "q4_linear_g128_pair"
    ):
        return None
    if group0 == 128 and os.getenv(
        "FLAGGEMS_Q4_FUSED_GDN_G128", "0"
    ).lower() not in {"1", "true", "on"}:
        return None

    rhs0 = getattr(first_layer, first_kernel.w_q_name)
    rhs1 = getattr(second_layer, second_kernel.w_q_name)
    def apply(x: torch.Tensor) -> torch.Tensor | None:
        if (
            x.device.type != "cpu"
            or x.shape[-1] != k0
            or x.numel() != k0
        ):
            return None
        source_dtype = x.dtype
        x_bf16 = x if source_dtype == torch.bfloat16 else x.to(torch.bfloat16)
        if group0 == 128:
            output = torch.ops.triton_jit_cpu.q4_linear_g128_pair(
                x_bf16.contiguous(), rhs0, n0, rhs1, n1, k0
            )
        else:
            output = torch.ops.triton_jit_cpu.q4_linear_g32_asym_compact_pair(
                x_bf16.contiguous(), rhs0, n0, rhs1, n1, k0
            )
        return output if source_dtype == torch.bfloat16 else output.to(source_dtype)

    return apply


def apply_vllm_q4_g32_pair(
    first_layer: torch.nn.Module,
    second_layer: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor | None:
    """One-shot compatibility wrapper around the cached-pair preparation."""
    apply = prepare_vllm_q4_g32_pair(first_layer, second_layer)
    return None if apply is None else apply(x)


def _enable_vllm_dynamic4bit_g128() -> None:
    """Route vLLM compressed-tensors G128 Q4 through libtriton_jit.

    vLLM normally repacks these parameters for
    ``aten._dyn_quant_matmul_4bit`` (KleidiAI on Arm).  Keep the checkpoint's
    signed INT4 values and BF16 G128 scales, pack the ordinary-Triton ABI once
    at model load, and replace only the kernel's two lifecycle methods.  Other
    group sizes and quantization formats retain vLLM's native implementation.
    """
    from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
        Dynamic4bitLinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils import replace_parameter
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    if getattr(
        Dynamic4bitLinearKernel,
        "_flag_gems_libtriton_jit_g128_enabled",
        False,
    ):
        return

    original_process = Dynamic4bitLinearKernel.process_weights_after_loading
    original_apply = Dynamic4bitLinearKernel.apply_weights
    from compressed_tensors.compressors.pack_quantized.helpers import (
        unpack_from_int32,
    )
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        config = self.config
        if config.group_size not in {32, 128} or config.zero_points:
            return original_process(self, layer)

        quantized = getattr(layer, self.w_q_name)
        scales = getattr(layer, self.w_s_name)
        k, n = config.partition_weight_shape
        packed_checkpoint = (
            quantized.dtype == torch.int32
            and quantized.shape == (n, k // 8)
        )
        if packed_checkpoint:
            weight_shape = getattr(layer, "weight_shape", None)
            if weight_shape is not None:
                loaded_shape = tuple(int(v) for v in weight_shape.tolist())
                # Stacked vLLM projections concatenate the packed q/k/v or
                # gate/up tensors, while this two-element auxiliary parameter
                # is overwritten by one component's original shape.  The
                # packed tensor itself is authoritative for total N; the
                # auxiliary shape can still catch a wrong K or impossible N.
                if (
                    len(loaded_shape) != 2
                    or loaded_shape[1] != k
                    or not 0 < loaded_shape[0] <= n
                ):
                    raise RuntimeError(
                        "unexpected compressed-tensors packed-Q4 weight_shape: "
                        f"loaded={loaded_shape}, fused_expected={(n, k)}"
                    )
            quantized_nk = unpack_from_int32(
                quantized.detach(), 4, torch.Size((n, k))
            ).contiguous()
        else:
            quantized_nk = quantized
        if (
            quantized_nk.dtype != torch.int8
            or quantized_nk.shape != (n, k)
            or scales.shape != (n, k // config.group_size)
        ):
            raise RuntimeError(
                "unexpected vLLM compressed-tensors grouped-Q4 parameter shape"
            )
        if os.getenv("FLAGGEMS_Q4_VLLM_NATIVE", "0").lower() in {
            "1", "true", "on"
        }:
            if packed_checkpoint:
                replace_parameter(
                    layer,
                    self.w_q_name,
                    torch.nn.Parameter(quantized_nk, requires_grad=False),
                )
                setattr(layer, "weight_shape", None)
            return original_process(self, layer)
        if config.group_size == 128:
            rhs = pack_rhs_qsi4c128p_asym(
                quantized_nk,
                scales.detach().to(torch.bfloat16).contiguous(),
            )
        else:
            rhs = pack_rhs_qsi4c32p_asym_compact(
                quantized_nk,
                scales.detach().to(torch.bfloat16).contiguous(),
            )
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(rhs, requires_grad=False),
        )
        # CompressedTensorsW4A8Int registers the checkpoint tensor under both
        # `weight` and `weight_packed`. Replacing only `weight_packed` leaves
        # the full [N,K] INT8 source alias alive even though each value holds
        # only one signed INT4 code. For Qwen3.6-27B that stale alias retains
        # 22.68 GiB in addition to the 12.77 GiB compact runtime layout.
        # Inference uses `weight_packed` through this kernel, so the `weight`
        # alias can be released after packing. Retaining it is an explicit
        # debugging option; production defaults to the compact runtime copy.
        release_source = os.getenv(
            "FLAGGEMS_Q4_RELEASE_SOURCE_WEIGHTS", "1"
        ).lower() in {"1", "true", "on"}
        source_alias = getattr(layer, "weight", None)
        if (
            release_source
            and self.w_q_name != "weight"
            and source_alias is not None
            and source_alias.untyped_storage().data_ptr()
            == quantized.untyped_storage().data_ptr()
        ):
            replace_parameter(
                layer,
                "weight",
                torch.nn.Parameter(
                    torch.empty(0, dtype=quantized.dtype, device=quantized.device),
                    requires_grad=False,
                ),
            )
        # Store only serializable data on the kernel object.  The previous
        # function-valued shortcut made Torch AOT deepcopy fail.
        if _USE_VLLM_FAST_APPLY:
            self._flag_gems_q4_prepared_rhs = getattr(layer, self.w_q_name)
        setattr(layer, self.w_s_name, None)
        if packed_checkpoint:
            setattr(layer, "weight_shape", None)
        self._flag_gems_libtriton_jit_q4_shape = (
            n, k, config.group_size
        )
        _STATS[
            "prepared_g128_linears"
            if config.group_size == 128
            else "prepared_g32_linears"
        ] += 1

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prepared_rhs = getattr(self, "_flag_gems_q4_prepared_rhs", None)
        prepared_shape = getattr(
            self, "_flag_gems_libtriton_jit_q4_shape", None
        )
        if (
            _USE_VLLM_FAST_APPLY
            and prepared_rhs is not None
            and prepared_shape is not None
            and bias is None
            and x.dtype == torch.bfloat16
            and x.device.type == "cpu"
            and x.is_contiguous()
        ):
            n, k, group_size = prepared_shape
            if group_size == 128:
                return torch.ops.triton_jit_cpu.q4_linear_g128(
                    x, prepared_rhs, n, k
                )
            return torch.ops.triton_jit_cpu.q4_linear_g32_asym_compact(
                x, prepared_rhs, n, k
            )
        shape = prepared_shape
        if shape is None:
            return original_apply(self, layer, x, bias)
        n, k, group_size = shape
        source_dtype = x.dtype
        x_bf16 = x if source_dtype == torch.bfloat16 else x.to(torch.bfloat16)
        x_bf16 = x_bf16.contiguous()
        rhs = getattr(layer, self.w_q_name)
        if group_size == 128:
            output = torch.ops.triton_jit_cpu.q4_linear_g128(
                x_bf16, rhs, n, k
            )
        else:
            output = torch.ops.triton_jit_cpu.q4_linear_g32_asym_compact(
                x_bf16, rhs, n, k
            )
        if source_dtype != torch.bfloat16:
            output = output.to(source_dtype)
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output

    def apply_swiglu_weights(
        self,
        layer: torch.nn.Module,
        joined: torch.Tensor,
    ) -> torch.Tensor | None:
        shape = getattr(self, "_flag_gems_libtriton_jit_q4_shape", None)
        if shape is None:
            return None
        n, k, group_size = shape
        if (
            group_size not in {32, 128}
            or joined.dtype != torch.bfloat16
            or joined.device.type != "cpu"
            or not joined.is_contiguous()
            or joined.shape[-1] != 2 * k
        ):
            return None
        rhs = getattr(layer, self.w_q_name)
        if group_size == 128:
            if not hasattr(torch.ops.triton_jit_cpu, "q4_linear_g128_swiglu"):
                return None
            return torch.ops.triton_jit_cpu.q4_linear_g128_swiglu(
                joined, rhs, n, k
            )
        return torch.ops.triton_jit_cpu.q4_linear_g32_asym_compact_swiglu(
            joined, rhs, n, k
        )

    original_mlp_forward = Qwen2MoeMLP.forward
    use_fused_swiglu = os.getenv(
        "FLAGGEMS_Q4_FUSED_SWIGLU_DOWN", "1"
    ).lower() in {"1", "true", "on"}

    def qwen_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        kernel = _vllm_dynamic4bit_kernel(self.down_proj)
        fused = None
        if (
            type(self)._flag_gems_fused_swiglu_active
            and self.expert_gate is None
            and self.down_proj.tp_size == 1
            and self.down_proj.bias is None
            and kernel is not None
        ):
            apply_swiglu = getattr(
                kernel, "_flag_gems_apply_swiglu_weights", None
            )
            if apply_swiglu is not None:
                fused = apply_swiglu(self.down_proj, gate_up)
        if fused is not None:
            return fused

        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        if self.expert_gate is not None:
            out = torch.nn.functional.sigmoid(self.expert_gate(x)[0]) * out
        return out

    Dynamic4bitLinearKernel.process_weights_after_loading = (
        process_weights_after_loading
    )
    Dynamic4bitLinearKernel.apply_weights = apply_weights
    Dynamic4bitLinearKernel._flag_gems_apply_swiglu_weights = (
        apply_swiglu_weights
    )
    Dynamic4bitLinearKernel._flag_gems_libtriton_jit_g128_enabled = True
    Dynamic4bitLinearKernel._flag_gems_libtriton_jit_g128_original_process = (
        original_process
    )
    Dynamic4bitLinearKernel._flag_gems_libtriton_jit_g128_original_apply = (
        original_apply
    )
    if not getattr(Qwen2MoeMLP, "_flag_gems_fused_swiglu_enabled", False):
        Qwen2MoeMLP.forward = qwen_mlp_forward
        Qwen2MoeMLP._flag_gems_fused_swiglu_enabled = True
        Qwen2MoeMLP._flag_gems_fused_swiglu_original_forward = (
            original_mlp_forward
        )
    Qwen2MoeMLP._flag_gems_fused_swiglu_active = use_fused_swiglu


def _enable_vllm_dynamic_int8() -> None:
    """Route eligible compressed-tensors W8A8 linears through libtriton_jit."""
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
        CPUInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils import replace_parameter

    if getattr(
        CPUInt8ScaledMMLinearKernel,
        "_flag_gems_libtriton_jit_w8_enabled",
        False,
    ):
        return

    original_process = CPUInt8ScaledMMLinearKernel.process_weights_after_loading
    original_apply = CPUInt8ScaledMMLinearKernel.apply_weights

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        config = self.config
        w_q_name, w_s_name, _, _, _ = self.layer_param_names
        weight = getattr(layer, w_q_name)
        eligible = (
            config.is_channelwise
            and not config.is_static_input_scheme
            and config.input_symmetric
            and weight.ndim == 2
            and weight.dtype == torch.int8
            and weight.shape[0] % 4 == 0
            and weight.shape[1] % 32 == 0
        )
        if not eligible:
            return original_process(self, layer)

        n, k = weight.shape
        scale = getattr(layer, w_s_name).reshape(-1).to(torch.float32)
        if scale.numel() != n:
            raise RuntimeError("unexpected vLLM W8 per-channel scale shape")
        weight_nk = weight.detach().contiguous()
        rhs = pack_rhs_w8_symmetric(weight_nk, scale)
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(rhs, requires_grad=False),
        )
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(scale.contiguous(), requires_grad=False),
        )
        # Match the Q4 fast path: retain only serializable prepared data on
        # the quant kernel so AOT deepcopy remains valid while hot calls avoid
        # resolving the packed parameter through the layer on every linear.
        if _USE_VLLM_FAST_APPLY:
            self._flag_gems_w8_prepared_rhs = getattr(layer, w_q_name)
        self._flag_gems_libtriton_jit_w8_shape = (n, k)
        _STATS["prepared_w8_linears"] += 1

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prepared_rhs = getattr(self, "_flag_gems_w8_prepared_rhs", None)
        shape = getattr(self, "_flag_gems_libtriton_jit_w8_shape", None)
        if (
            _USE_VLLM_FAST_APPLY
            and prepared_rhs is not None
            and shape is not None
            and bias is None
            and x.dtype == torch.bfloat16
            and x.device.type == "cpu"
            and x.is_contiguous()
        ):
            n, k = shape
            return torch.ops.triton_jit_cpu.w8_linear_kai(
                x, prepared_rhs, n, k
            )
        if shape is None:
            return original_apply(self, layer, x, bias)
        n, k = shape
        w_q_name, _, _, _, _ = self.layer_param_names
        source_dtype = x.dtype
        x_bf16 = x if source_dtype == torch.bfloat16 else x.to(torch.bfloat16)
        output = torch.ops.triton_jit_cpu.w8_linear_kai(
            x_bf16.contiguous(),
            getattr(layer, w_q_name),
            n,
            k,
        )
        if source_dtype != torch.bfloat16:
            output = output.to(source_dtype)
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output

    CPUInt8ScaledMMLinearKernel.process_weights_after_loading = (
        process_weights_after_loading
    )
    CPUInt8ScaledMMLinearKernel.apply_weights = apply_weights
    CPUInt8ScaledMMLinearKernel._flag_gems_libtriton_jit_w8_enabled = True
    CPUInt8ScaledMMLinearKernel._flag_gems_libtriton_jit_w8_original_process = (
        original_process
    )
    CPUInt8ScaledMMLinearKernel._flag_gems_libtriton_jit_w8_original_apply = (
        original_apply
    )


def _arm_codegen_features() -> set[str]:
    """Return Arm features through either supported Triton-CPU API."""
    try:
        from triton._C.libtriton import llvm
        from triton.backends.cpu.target_info import (
            supplement_aarch64_features,
        )
    except ModuleNotFoundError:
        # Compatibility with the older FlagTree/Triton-CPU tree.
        from triton.language.extra.cpu.cpu_features import features

        return set(features())

    detected = supplement_aarch64_features(llvm.get_cpu_features())
    # Preserve the disable-only controls used by the older detector.  They
    # must never be able to enable an instruction missing from the host.
    if os.getenv("TLE_CPU_DISABLE_DOTPROD", "0") != "0":
        detected -= {"dotprod", "i8mm"}
    if os.getenv("TLE_CPU_DISABLE_I8MM", "0") != "0":
        detected.discard("i8mm")
    return detected


def enable_vllm_q4_codegen(
    verbose: bool = True, runtime: str = "python"
) -> None:
    """Install the production vLLM Q4/Q8 libtriton_jit/codegen router."""
    if runtime not in {"python", "libtriton_jit"}:
        raise ValueError("Q4 codegen runtime must be 'python' or 'libtriton_jit'")
    if runtime == "libtriton_jit":
        from flag_gems.csrc.arm import configure_runtime

        try:
            library = configure_runtime()
        except FileNotFoundError as error:
            raise RuntimeError(str(error)) from error
        torch.ops.load_library(os.path.realpath(library))
        required_ops = (
            "q4_linear",
            "q4_linear_g32_asym",
            "q4_linear_g32_asym_compact",
            "q4_linear_g32_asym_compact_swiglu",
            "q4_linear_g128",
            "w8_linear",
            "w8_linear_kai",
        )
        if not all(
            hasattr(torch.ops.triton_jit_cpu, name) for name in required_ops
        ):
            raise RuntimeError("libtriton_jit ARM operator registration failed")
        _enable_vllm_dynamic4bit_g128()
        _enable_vllm_dynamic_int8()
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("ARM Q4 codegen router requires an AArch64 host")

    required_features = {"dotprod", "i8mm"}
    missing_features = required_features - _arm_codegen_features()
    if missing_features:
        missing = ", ".join(sorted(missing_features))
        raise RuntimeError(
            f"ARM Q4 codegen router requires CPU features: {missing}"
        )
    import vllm.model_executor.layers.utils as layer_utils
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    if getattr(layer_utils, "_flag_gems_q4_codegen_enabled", False):
        active = getattr(
            layer_utils, "_flag_gems_q4_codegen_runtime", "python"
        )
        if active != runtime:
            raise RuntimeError(
                f"ARM Q4 router already enabled with runtime={active}"
            )
        return

    original_dispatch = layer_utils.dispatch_cpu_unquantized_gemm
    include_q4_lm_head = os.getenv("FL_INT4_LMHEAD", "0") == "1"
    include_w8_lm_head = os.getenv("FL_INT8_LMHEAD", "0") == "1"
    online_w8 = os.getenv("FL_CPU_INT8", "0").lower() in {
        "1", "true", "on"
    }
    if include_q4_lm_head and include_w8_lm_head:
        raise RuntimeError(
            "FL_INT4_LMHEAD and FL_INT8_LMHEAD are mutually exclusive"
        )
    if (include_w8_lm_head or online_w8) and runtime != "libtriton_jit":
        raise RuntimeError("production W8 routing requires runtime=libtriton_jit")
    strict = os.getenv("FLAGGEMS_ARM_Q4_STRICT", "1") != "0"

    def dispatch(layer, remove_weight):
        weight = getattr(layer, "weight", None)
        prefix = getattr(layer, "prefix", "") or ""
        is_lm_head = isinstance(layer, ParallelLMHead)
        is_embedding = (
            isinstance(layer, VocabParallelEmbedding) and not is_lm_head
        )
        eligible_base = (
            weight is not None
            and not weight.is_meta
            and weight.device.type == "cpu"
            and weight.is_floating_point()
            and weight.ndim == 2
            and weight.shape[1] % BLOCK_LENGTH == 0
            and not is_embedding
        )
        use_w8 = eligible_base and (
            (is_lm_head and include_w8_lm_head)
            or (not is_lm_head and online_w8)
        )
        use_q4 = eligible_base and not use_w8 and weight.shape[0] % 4 == 0 and (
            (is_lm_head and include_q4_lm_head)
            or (not is_lm_head and not online_w8)
        )
        if use_w8 and weight.shape[0] % 4:
            use_w8 = False
        if use_w8 or use_q4:
            try:
                n, k = weight.shape
                if use_w8:
                    rhs = prepare_w8_weight_kai(weight.detach())
                    layer.cpu_linear = _make_vllm_w8_linear(
                        rhs, n, k
                    )
                    _STATS["prepared_online_w8_linears"] += 1
                    if is_lm_head:
                        _STATS["prepared_w8_lm_heads"] += 1
                else:
                    asymmetric = is_lm_head and runtime == "libtriton_jit"
                    rhs_codegen = (
                        prepare_weight_asym(weight.detach())
                        if asymmetric
                        else prepare_weight(weight.detach())
                    )
                    layer.cpu_linear = _make_vllm_linear(
                        rhs_codegen,
                        n,
                        k,
                        runtime=runtime,
                        asymmetric=asymmetric,
                    )
                    if is_lm_head:
                        _STATS["prepared_q4_lm_heads"] += 1
                if remove_weight:
                    layer.weight = torch.nn.Parameter(
                        torch.empty(0), requires_grad=False
                    )
                elif is_lm_head and os.getenv(
                    "FLAGGEMS_RELEASE_BF16_LM_HEAD", "1"
                ).lower() in {"1", "true", "on"}:
                    # ParallelLMHead shares the generic embedding method, so
                    # vLLM requests remove_weight=False even though this layer
                    # is never used for token lookup. The prepared W8/Q4
                    # closure ignores the original BF16 argument; retaining it
                    # costs 2.37 GiB for Qwen3.6-27B.
                    layer.weight = torch.nn.Parameter(
                        torch.empty(0, dtype=weight.dtype, device=weight.device),
                        requires_grad=False,
                    )
                _STATS["prepared_linears"] += 1
                return
            except Exception as exc:
                message = (
                    f"failed to prepare ARM {'W8' if use_w8 else 'Q4'} "
                    f"codegen weight {prefix} "
                    f"{tuple(weight.shape)}"
                )
                if strict:
                    raise RuntimeError(message) from exc
                logger.warning("%s; falling back to vLLM BF16: %s", message, exc)
        return original_dispatch(layer, remove_weight)

    layer_utils.dispatch_cpu_unquantized_gemm = dispatch
    layer_utils._flag_gems_q4_codegen_enabled = True
    layer_utils._flag_gems_q4_codegen_runtime = runtime
    if verbose:
        print(
            "[flag_gems] ARM quantized Triton router enabled "
            f"(runtime={runtime}, M<4=Triton SDOT, "
            "prefill=Triton target block+tail, body=checkpoint G128/W8, "
            "head=G32/W8 by selector)",
            flush=True,
        )


def stats() -> dict[str, int]:
    return dict(_STATS)
