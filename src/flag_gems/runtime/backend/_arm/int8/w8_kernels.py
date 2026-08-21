"""Ordinary-Triton W8 entry points used by the C++ libtriton_jit router.

The ``*_qai8dxp_*`` entry points implement the deployment guide's exact
KleidiAI data contract.  They are compiler-visible Triton programs, not calls
to a KleidiAI or TLE runtime implementation.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.runtime.backend._arm.int8.tle_int8_linear import (
    _pack_lhs_w8_i8mm_kai_kernel,
    _pack_lhs_w8_i8mm_kai_vllm_trunc_kernel,
    _quantize_bf16_w8_rne_kernel,
    _quantize_bf16_w8_vllm_trunc_kernel,
    _w8_decode_sdot_kernel,
    _w8_prefill_i8mm_kai_kernel,
    _w8_prefill_i8mm_kai_m12_kernel,
    _w8_prefill_i8mm_kai_short_tail_kernel,
)


@triton.jit
def _pack_lhs_qai8dxp_bf16_kernel(
    x_ptr,
    packed_ptr,
    M: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    K: tl.constexpr,
    MR: tl.constexpr,
):
    """Pack BF16 rows with KleidiAI's scalar-row ``qai8dxp`` schedule.

    Production decode specializes this to M=MR=1.  Exposing every shape value
    as a constexpr also gives the generated compute function the same ABI as
    the standalone byte-for-byte KleidiAI comparator.
    """
    row = tl.program_id(0)
    row_ok = row < M
    source_row = tl.minimum(row, M - 1)
    row_base = x_ptr + source_row * STRIDE_XM

    lanes32 = tl.arange(0, 32)
    row_min = tl.zeros((1,), tl.float32)
    row_max = tl.zeros((1,), tl.float32)
    for off in tl.range(0, K, 32, loop_unroll_factor=1):
        values = tl.load(row_base + off + lanes32).to(tl.float32)
        values = tl.where(row_ok, values, 0.0)
        row_min = tl.minimum(row_min, tl.min(values, axis=0))
        row_max = tl.maximum(row_max, tl.max(values, axis=0))

    quant_multiplier = tl.where(
        row_min == row_max, 1.0, 255.0 / (row_max - row_min)
    )
    reciprocal_scale = tl.where(
        quant_multiplier != 0.0, 1.0 / quant_multiplier, 0.0
    )
    descaled_min = row_min * quant_multiplier
    descaled_max = row_max * quant_multiplier
    zero_point = tl.where(
        -128.0 + descaled_min + 127.0 + descaled_max > 0.0,
        -128.0 - descaled_min,
        127.0 - descaled_max,
    )
    zero_point = tl.minimum(tl.maximum(zero_point, -128.0), 127.0)
    zero_point_i32 = libdevice.rint(zero_point).to(tl.int32)

    panel = row // MR
    panel_row = row % MR
    panel_stride: tl.constexpr = MR * (K + 8)
    lanes8 = tl.arange(0, 8)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(row_base + off + lanes8).to(tl.float32)
        values = tl.where(row_ok, values, 0.0)
        quantized = (
            libdevice.rint(values * quant_multiplier).to(tl.int32)
            + zero_point_i32
        )
        quantized = tl.minimum(tl.maximum(quantized, -128), 127).to(tl.int8)
        packed_offset = (
            panel * panel_stride
            + (off // 8) * MR * 8
            + panel_row * 8
        )
        tl.store(packed_ptr + packed_offset + lanes8, quantized)

    metadata = packed_ptr + panel * panel_stride + MR * K
    tl.store(
        metadata.to(tl.pointer_type(tl.int32))
        + panel_row
        + tl.arange(0, 1),
        -zero_point_i32,
    )
    tl.store(
        (metadata + MR * 4).to(tl.pointer_type(tl.float32))
        + panel_row
        + tl.arange(0, 1),
        reciprocal_scale,
    )


@triton.jit
def _pack_lhs_qai8dxp_bf16_mr4_kernel(
    x_ptr,
    packed_ptr,
    M: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    K: tl.constexpr,
):
    """Pack one (possibly padded) MR4 BF16 panel into ``qai8dxp``."""
    panel = tl.program_id(0)
    rows = panel * 4 + tl.arange(0, 4)
    source_rows = tl.minimum(rows, M - 1)
    lanes8 = tl.arange(0, 8)
    row_min_lanes = tl.zeros((4, 8), tl.float32)
    row_max_lanes = tl.zeros((4, 8), tl.float32)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(
            x_ptr
            + source_rows[:, None] * STRIDE_XM
            + off
            + lanes8[None, :]
        ).to(tl.float32)
        row_min_lanes = tl.minimum(row_min_lanes, values)
        row_max_lanes = tl.maximum(row_max_lanes, values)
    row_min = tl.min(row_min_lanes, axis=1)
    row_max = tl.max(row_max_lanes, axis=1)

    quant_multiplier = tl.where(
        row_min == row_max, 1.0, 255.0 / (row_max - row_min)
    )
    scales = tl.where(
        quant_multiplier == 0.0, 0.0, 1.0 / quant_multiplier
    )
    descaled_min = row_min * quant_multiplier
    descaled_max = row_max * quant_multiplier
    zero_points = tl.where(
        -128.0 + descaled_min + 127.0 + descaled_max > 0.0,
        -128.0 - descaled_min,
        127.0 - descaled_max,
    )
    zero_points = tl.minimum(tl.maximum(zero_points, -128.0), 127.0)
    zero_points = libdevice.rint(zero_points).to(tl.int32)

    panel_stride: tl.constexpr = 4 * K + 32
    panel_base = packed_ptr + panel * panel_stride
    physical_lanes = tl.arange(0, 32)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(
            x_ptr
            + source_rows[:, None] * STRIDE_XM
            + off
            + lanes8[None, :]
        ).to(tl.float32)
        # Match the BF16 NEON packer's vcvtnq_s32_f32 conversion.
        rounded = libdevice.rint(
            values * quant_multiplier[:, None]
        ).to(tl.int32)
        # KleidiAI narrows the rounded int32 values before adding the zero
        # point.  The finite asymmetric range is within int16, so this is exact
        # and exposes the same halfword add/clamp/narrow sequence to LLVM.
        quantized = rounded.to(tl.int16) + zero_points.to(tl.int16)[:, None]
        quantized = tl.minimum(tl.maximum(quantized, -128), 127).to(tl.int8)
        tl.store(
            panel_base + (off // 8) * 32 + physical_lanes,
            quantized.reshape((32,)),
        )

    metadata = panel_base + 4 * K
    tl.store(
        metadata.to(tl.pointer_type(tl.int32)) + tl.arange(0, 4),
        -zero_points,
    )
    tl.store(
        (metadata + 16).to(tl.pointer_type(tl.float32)) + tl.arange(0, 4),
        scales,
    )


@triton.jit
def _w8_qai8dxp_decode_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Exact-KAI N4/K8 W8 decode graph, partitioned across CPU programs."""
    partition = tl.program_id(0)
    partitions = tl.num_programs(0)
    tile_count = range_end - range_begin
    tiles_per_partition = (tile_count + partitions - 1) // partitions
    local_begin = range_begin + partition * tiles_per_partition
    local_end = tl.minimum(local_begin + tiles_per_partition, range_end)

    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 16)
    output_lanes = tl.arange(0, 4)
    x_offsets = tl.arange(0, 8)

    for block in range(local_begin, local_end):
        partial01 = tl.zeros((4,), dtype=tl.int32)
        partial23 = tl.zeros((4,), dtype=tl.int32)
        rhs_tile_ptr = rhs_packed_ptr + block * rhs_stride
        lhs_chunk_ptr = lhs_packed_ptr
        rhs_chunk_ptr = rhs_tile_ptr
        for _ in tl.range(0, K // 32, loop_unroll_factor=UNROLL):
            for sub in tl.static_range(0, 4):
                group_ptr = rhs_chunk_ptr + sub * 32
                weight01 = tl.load(group_ptr + q_offsets).reshape((4, 4))
                weight23 = tl.load(group_ptr + 16 + q_offsets).reshape((4, 4))
                x = tl.load(
                    lhs_chunk_ptr + sub * 8 + x_offsets
                ).reshape((2, 4))
                # Keeping K8 intact lets the Arm backend select LD1R+SDOT.
                x_repeated = tl.join(x, x).permute(2, 0, 1).reshape((4, 4))
                partial01 += tl.sum(
                    weight01.to(tl.int32) * x_repeated.to(tl.int32), axis=1
                )
                partial23 += tl.sum(
                    weight23.to(tl.int32) * x_repeated.to(tl.int32), axis=1
                )
            lhs_chunk_ptr += 32
            rhs_chunk_ptr += 128

        partial = tl.join(partial01, partial23).permute(1, 0).reshape((4, 2))
        dot = tl.sum(partial, axis=1)
        metadata = (rhs_tile_ptr + 4 * K).to(tl.pointer_type(tl.int32))
        rhs_sum = tl.load(metadata + output_lanes)
        rhs_scale = tl.load(
            (rhs_tile_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
            + output_lanes
        )
        bias = tl.load(
            (rhs_tile_ptr + 4 * K + 32).to(tl.pointer_type(tl.float32))
            + output_lanes
        )
        corrected = dot + rhs_sum * lhs_offset
        combined_scale = lhs_scale * rhs_scale
        result = corrected.to(tl.float32) * combined_scale + bias
        tl.store(
            out_ptr + block * 4 + output_lanes,
            result.to(tl.bfloat16),
        )


@triton.jit
def _w8_qai8dxp_decode_stealing_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    counter_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    STEAL_CHUNK: tl.constexpr,
):
    """Exact-KAI W8 GEMV with dynamic N4 work assignment."""
    lhs_offset = tl.load(
        (lhs_packed_ptr + K).to(tl.pointer_type(tl.int32))
    )
    lhs_scale = tl.load(
        (lhs_packed_ptr + K + 4).to(tl.pointer_type(tl.float32))
    )
    rhs_stride: tl.constexpr = 4 * K + 48
    q_offsets = tl.arange(0, 16)
    output_lanes = tl.arange(0, 4)
    x_offsets = tl.arange(0, 8)

    local_begin = range_begin + tl.atomic_add(counter_ptr, STEAL_CHUNK)
    while local_begin < range_end:
        local_end = tl.minimum(local_begin + STEAL_CHUNK, range_end)
        for block in range(local_begin, local_end):
            partial01 = tl.zeros((4,), dtype=tl.int32)
            partial23 = tl.zeros((4,), dtype=tl.int32)
            rhs_tile_ptr = rhs_packed_ptr + block * rhs_stride
            lhs_chunk_ptr = lhs_packed_ptr
            rhs_chunk_ptr = rhs_tile_ptr
            for _ in tl.range(0, K // 32, loop_unroll_factor=UNROLL):
                for sub in tl.static_range(0, 4):
                    group_ptr = rhs_chunk_ptr + sub * 32
                    weight01 = tl.load(group_ptr + q_offsets).reshape((4, 4))
                    weight23 = tl.load(
                        group_ptr + 16 + q_offsets
                    ).reshape((4, 4))
                    x = tl.load(
                        lhs_chunk_ptr + sub * 8 + x_offsets
                    ).reshape((2, 4))
                    x_repeated = tl.join(x, x).permute(2, 0, 1).reshape((4, 4))
                    partial01 += tl.sum(
                        weight01.to(tl.int32) * x_repeated.to(tl.int32),
                        axis=1,
                    )
                    partial23 += tl.sum(
                        weight23.to(tl.int32) * x_repeated.to(tl.int32),
                        axis=1,
                    )
                lhs_chunk_ptr += 32
                rhs_chunk_ptr += 128

            partial = tl.join(partial01, partial23).permute(1, 0).reshape((4, 2))
            dot = tl.sum(partial, axis=1)
            metadata = (rhs_tile_ptr + 4 * K).to(tl.pointer_type(tl.int32))
            rhs_sum = tl.load(metadata + output_lanes)
            rhs_scale = tl.load(
                (rhs_tile_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
                + output_lanes
            )
            bias = tl.load(
                (rhs_tile_ptr + 4 * K + 32).to(tl.pointer_type(tl.float32))
                + output_lanes
            )
            corrected = dot + rhs_sum * lhs_offset
            result = (
                corrected.to(tl.float32) * (lhs_scale * rhs_scale) + bias
            )
            tl.store(
                out_ptr + block * 4 + output_lanes,
                result.to(tl.bfloat16),
            )
        local_begin = range_begin + tl.atomic_add(counter_ptr, STEAL_CHUNK)


@triton.jit
def _w8_qai8dxp_prefill_i8mm_tile(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PID_M_OVERRIDE: tl.constexpr,
    PID_N_OVERRIDE: tl.constexpr,
):
    """Exact-KAI asymmetric M16/N4 prefill, lowered to target I8MM."""
    pid_m = PID_M_OVERRIDE
    pid_n = PID_N_OVERRIDE
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 4)
    panel_offsets = tl.arange(0, 128)
    lhs_panel_stride: tl.constexpr = 4 * K + 32
    rhs_panel_stride: tl.constexpr = 4 * K + 48
    accumulator = tl.zeros((BLOCK_M, 4), tl.int32)

    for chunk in range(0, K // 32):
        lhs_base = pid_m * 4 * lhs_panel_stride + chunk * 128
        lhs0 = tl.load(
            lhs_packed_ptr + lhs_base + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs1 = tl.load(
            lhs_packed_ptr + lhs_base + lhs_panel_stride + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs2 = tl.load(
            lhs_packed_ptr + lhs_base + 2 * lhs_panel_stride + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs3 = tl.load(
            lhs_packed_ptr + lhs_base + 3 * lhs_panel_stride + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs01 = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        lhs23 = tl.join(lhs2, lhs3).permute(2, 0, 1).reshape((8, 32))
        lhs = tl.join(lhs01, lhs23).permute(2, 0, 1).reshape((16, 32))
        rhs = tl.load(
            rhs_packed_ptr
            + pid_n * rhs_panel_stride
            + chunk * 128
            + panel_offsets
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)

    meta_lanes = tl.arange(0, 4)
    lhs_meta = pid_m * 4 * lhs_panel_stride + 4 * K
    offset0 = tl.load(
        (lhs_packed_ptr + lhs_meta).to(tl.pointer_type(tl.int32)) + meta_lanes
    )
    offset1 = tl.load(
        (lhs_packed_ptr + lhs_meta + lhs_panel_stride).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    offset2 = tl.load(
        (lhs_packed_ptr + lhs_meta + 2 * lhs_panel_stride).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    offset3 = tl.load(
        (lhs_packed_ptr + lhs_meta + 3 * lhs_panel_stride).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    offset01 = tl.join(offset0, offset1).permute(1, 0).reshape((8,))
    offset23 = tl.join(offset2, offset3).permute(1, 0).reshape((8,))
    lhs_offset = tl.join(offset01, offset23).permute(1, 0).reshape((16,))
    scale0 = tl.load(
        (lhs_packed_ptr + lhs_meta + 16).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    scale1 = tl.load(
        (lhs_packed_ptr + lhs_meta + lhs_panel_stride + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    scale2 = tl.load(
        (lhs_packed_ptr + lhs_meta + 2 * lhs_panel_stride + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    scale3 = tl.load(
        (lhs_packed_ptr + lhs_meta + 3 * lhs_panel_stride + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    scale01 = tl.join(scale0, scale1).permute(1, 0).reshape((8,))
    scale23 = tl.join(scale2, scale3).permute(1, 0).reshape((8,))
    lhs_scale = tl.join(scale01, scale23).permute(1, 0).reshape((16,))

    rhs_meta = pid_n * rhs_panel_stride + 4 * K
    rhs_sum = tl.load(
        (rhs_packed_ptr + rhs_meta).to(tl.pointer_type(tl.int32)) + cols
    )
    rhs_scale = tl.load(
        (rhs_packed_ptr + rhs_meta + 16).to(tl.pointer_type(tl.float32)) + cols
    )
    bias = tl.load(
        (rhs_packed_ptr + rhs_meta + 32).to(tl.pointer_type(tl.float32)) + cols
    )
    corrected = accumulator + lhs_offset[:, None] * rhs_sum[None, :]
    combined_scale = lhs_scale[:, None] * rhs_scale[None, :]
    result = corrected.to(tl.float32) * combined_scale + bias[None, :]
    output_rows = pid_m * BLOCK_M + rows
    output_cols = pid_n * 4 + cols
    tl.store(
        out_ptr + output_rows[:, None] * N + output_cols[None, :],
        result.to(tl.bfloat16),
    )


@triton.jit
def _w8_qai8dxp_prefill_i8mm_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    _w8_qai8dxp_prefill_i8mm_tile(
        lhs_packed_ptr,
        rhs_packed_ptr,
        out_ptr,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        PID_M_OVERRIDE=tl.program_id(0),
        PID_N_OVERRIDE=tl.program_id(1),
    )


@triton.jit
def _w8_qai8dxp_prefill_stealing_n_i8mm_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    counter_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    M_TILES: tl.constexpr,
    STEAL_CHUNK: tl.constexpr,
):
    """Dynamically distribute coarse N4 stripes across CPU workers."""
    counter_ptr = counter_ptr.to(tl.pointer_type(tl.int32))
    n_tiles: tl.constexpr = N // 4
    n_begin = tl.atomic_add(counter_ptr, STEAL_CHUNK)
    while n_begin < n_tiles:
        n_end = tl.minimum(n_begin + STEAL_CHUNK, n_tiles)
        for pid_n in range(n_begin, n_end):
            for pid_m in tl.range(0, M_TILES, loop_unroll_factor=1):
                _w8_qai8dxp_prefill_i8mm_tile(
                    lhs_packed_ptr,
                    rhs_packed_ptr,
                    out_ptr,
                    N=N,
                    K=K,
                    BLOCK_M=16,
                    PID_M_OVERRIDE=pid_m,
                    PID_N_OVERRIDE=pid_n,
                )
        n_begin = tl.atomic_add(counter_ptr, STEAL_CHUNK)


@triton.jit
def _w8_qai8dxp_prefill_short_tail_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Exact-KAI asymmetric M4/M8 prefill without M16 padded work."""
    pid_n = tl.program_id(1)
    cols = tl.arange(0, 4)
    panel_offsets = tl.arange(0, 128)
    meta_lanes = tl.arange(0, 4)
    lhs_panel_stride: tl.constexpr = 4 * K + 32
    rhs_panel_stride: tl.constexpr = 4 * K + 48
    accumulator = tl.zeros((BLOCK_M, 4), tl.int32)

    for chunk in range(0, K // 32):
        rhs = tl.load(
            rhs_packed_ptr
            + pid_n * rhs_panel_stride
            + chunk * 128
            + panel_offsets
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        lhs0 = tl.load(
            lhs_packed_ptr + chunk * 128 + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        if BLOCK_M == 4:
            lhs = lhs0
        else:
            lhs1 = tl.load(
                lhs_packed_ptr
                + lhs_panel_stride
                + chunk * 128
                + panel_offsets
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)

    rhs_meta = pid_n * rhs_panel_stride + 4 * K
    rhs_sum = tl.load(
        (rhs_packed_ptr + rhs_meta).to(tl.pointer_type(tl.int32)) + cols
    )
    rhs_scale = tl.load(
        (rhs_packed_ptr + rhs_meta + 16).to(tl.pointer_type(tl.float32)) + cols
    )
    bias = tl.load(
        (rhs_packed_ptr + rhs_meta + 32).to(tl.pointer_type(tl.float32)) + cols
    )
    offset0 = tl.load(
        (lhs_packed_ptr + 4 * K).to(tl.pointer_type(tl.int32)) + meta_lanes
    )
    scale0 = tl.load(
        (lhs_packed_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    if BLOCK_M == 4:
        lhs_offset = offset0
        lhs_scale = scale0
    else:
        offset1 = tl.load(
            (lhs_packed_ptr + lhs_panel_stride + 4 * K).to(
                tl.pointer_type(tl.int32)
            )
            + meta_lanes
        )
        scale1 = tl.load(
            (lhs_packed_ptr + lhs_panel_stride + 4 * K + 16).to(
                tl.pointer_type(tl.float32)
            )
            + meta_lanes
        )
        lhs_offset = tl.join(offset0, offset1).permute(1, 0).reshape((8,))
        lhs_scale = tl.join(scale0, scale1).permute(1, 0).reshape((8,))
    corrected = accumulator + lhs_offset[:, None] * rhs_sum[None, :]
    combined_scale = lhs_scale[:, None] * rhs_scale[None, :]
    result = corrected.to(tl.float32) * combined_scale + bias[None, :]
    rows = tl.arange(0, BLOCK_M)
    output_cols = pid_n * 4 + cols
    tl.store(
        out_ptr + rows[:, None] * N + output_cols[None, :],
        result.to(tl.bfloat16),
    )


@triton.jit
def _w8_qai8dxp_prefill_m12_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """Exact-KAI asymmetric M12 as one fused M8 plus M4 K loop."""
    pid_n = tl.program_id(1)
    cols = tl.arange(0, 4)
    panel_offsets = tl.arange(0, 128)
    meta_lanes = tl.arange(0, 4)
    lhs_panel_stride: tl.constexpr = 4 * K + 32
    rhs_panel_stride: tl.constexpr = 4 * K + 48
    accumulator8 = tl.zeros((8, 4), tl.int32)
    accumulator4 = tl.zeros((4, 4), tl.int32)

    for chunk in range(0, K // 32):
        rhs = tl.load(
            rhs_packed_ptr
            + pid_n * rhs_panel_stride
            + chunk * 128
            + panel_offsets
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        lhs0 = tl.load(
            lhs_packed_ptr + chunk * 128 + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs1 = tl.load(
            lhs_packed_ptr
            + lhs_panel_stride
            + chunk * 128
            + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs2 = tl.load(
            lhs_packed_ptr
            + 2 * lhs_panel_stride
            + chunk * 128
            + panel_offsets
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs8 = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        accumulator8 += tl.dot(lhs8, rhs, out_dtype=tl.int32)
        accumulator4 += tl.dot(lhs2, rhs, out_dtype=tl.int32)

    rhs_meta = pid_n * rhs_panel_stride + 4 * K
    rhs_sum = tl.load(
        (rhs_packed_ptr + rhs_meta).to(tl.pointer_type(tl.int32)) + cols
    )
    rhs_scale = tl.load(
        (rhs_packed_ptr + rhs_meta + 16).to(tl.pointer_type(tl.float32)) + cols
    )
    bias = tl.load(
        (rhs_packed_ptr + rhs_meta + 32).to(tl.pointer_type(tl.float32)) + cols
    )
    offset0 = tl.load(
        (lhs_packed_ptr + 4 * K).to(tl.pointer_type(tl.int32)) + meta_lanes
    )
    offset1 = tl.load(
        (lhs_packed_ptr + lhs_panel_stride + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    offset2 = tl.load(
        (lhs_packed_ptr + 2 * lhs_panel_stride + 4 * K).to(
            tl.pointer_type(tl.int32)
        )
        + meta_lanes
    )
    scale0 = tl.load(
        (lhs_packed_ptr + 4 * K + 16).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    scale1 = tl.load(
        (lhs_packed_ptr + lhs_panel_stride + 4 * K + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    scale2 = tl.load(
        (lhs_packed_ptr + 2 * lhs_panel_stride + 4 * K + 16).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    offset8 = tl.join(offset0, offset1).permute(1, 0).reshape((8,))
    scale8 = tl.join(scale0, scale1).permute(1, 0).reshape((8,))
    corrected8 = accumulator8 + offset8[:, None] * rhs_sum[None, :]
    corrected4 = accumulator4 + offset2[:, None] * rhs_sum[None, :]
    result8 = corrected8.to(tl.float32) * (
        scale8[:, None] * rhs_scale[None, :]
    ) + bias[None, :]
    result4 = corrected4.to(tl.float32) * (
        scale2[:, None] * rhs_scale[None, :]
    ) + bias[None, :]
    output_cols = pid_n * 4 + cols
    tl.store(
        out_ptr + tl.arange(0, 8)[:, None] * N + output_cols[None, :],
        result8.to(tl.bfloat16),
    )
    tl.store(
        out_ptr + (8 + meta_lanes)[:, None] * N + output_cols[None, :],
        result4.to(tl.bfloat16),
    )

__all__ = [
    "_pack_lhs_w8_i8mm_kai_kernel",
    "_pack_lhs_w8_i8mm_kai_vllm_trunc_kernel",
    "_quantize_bf16_w8_rne_kernel",
    "_quantize_bf16_w8_vllm_trunc_kernel",
    "_w8_decode_sdot_kernel",
    "_w8_prefill_i8mm_kai_kernel",
    "_w8_prefill_i8mm_kai_m12_kernel",
    "_w8_prefill_i8mm_kai_short_tail_kernel",
    "_pack_lhs_qai8dxp_bf16_kernel",
    "_pack_lhs_qai8dxp_bf16_mr4_kernel",
    "_w8_qai8dxp_decode_sdot_kernel",
    "_w8_qai8dxp_decode_stealing_sdot_kernel",
    "_w8_qai8dxp_prefill_i8mm_kernel",
    "_w8_qai8dxp_prefill_m12_kernel",
    "_w8_qai8dxp_prefill_short_tail_kernel",
]
