# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in packed W8A8 SME2 GEMM for the validated Apple 512-bit CPU target.

Input ABI v1: M16 K4-interleaved int8, then 16 negative zero-points (int32)
and 16 activation scales (float32). RHS: [N/16,K/128,32,16,4] int8.
Canonical codes/scales are not changed. A8 quantization is caller-owned.
Requires the full-K SME2 lowering and sme_vector_bits compiler option.
This explicit custom op does not replace aten.mm or globally enable FlagGems.
"""
import torch
import triton
import triton.language as tl
from flag_gems.utils import triton_lang_extension as tle


@triton.jit
def _output_bf16_bits(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where(value != value, 0x7FC0, rounded).to(tl.uint16)


@triton.jit
def _store_scaled(
    C,
    accumulator,
    negative_zp,
    activation_scale,
    WSum,
    WScale,
    rows,
    cols,
    M: tl.constexpr,
    N: tl.constexpr,
    BF16_OUTPUT: tl.constexpr = False,
):
    weight_sum = tl.load(WSum + cols)
    weight_scale = tl.load(WScale + cols)
    corrected = accumulator + negative_zp[:, None] * weight_sum[None, :]
    # Same ordering as KAI: integer zero-point correction, product of the
    # two FP32 scales, then multiplication with the converted integer result.
    scale = activation_scale[:, None] * weight_scale[None, :]
    result = corrected.to(tl.float32) * scale
    if BF16_OUTPUT == 2:
        result = result.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    elif BF16_OUTPUT:
        result = _output_bf16_bits(result)
    tl.store(C + rows[:, None] * N + cols[None, :], result, rows[:, None] < M)


@triton.jit
def _accumulator_row(accumulator, ROW: tl.constexpr):
    # Constant split/shuffle, with no gather or arithmetic reduction.
    lo, hi = tl.split(accumulator.reshape(2, 128).trans(1, 0))
    if ROW >= 8:
        half = hi
    else:
        half = lo
    lo, hi = tl.split(half.reshape(2, 64).trans(1, 0))
    if ROW % 8 >= 4:
        quarter = hi
    else:
        quarter = lo
    lo, hi = tl.split(quarter.reshape(2, 32).trans(1, 0))
    if ROW % 4 >= 2:
        pair = hi
    else:
        pair = lo
    lo, hi = tl.split(pair.reshape(2, 16).trans(1, 0))
    if ROW % 2:
        return hi
    else:
        return lo


@triton.jit
def _store_scaled_rows(
    C,
    accumulator,
    AInt,
    AFloat,
    WSum,
    WScale,
    mi,
    cols,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BF16_OUTPUT: tl.constexpr = False,
    PAIR_N: tl.constexpr = 0,
):
    # Emit one contiguous row at a time instead of materializing all broadcast
    # scale/zero-point arrays for an entire ZA tile in general registers.
    weight_sum = tl.load(WSum + cols)
    weight_scale = tl.load(WScale + cols)
    for row in tl.static_range(16):
        values = _accumulator_row(accumulator, row)
        meta = mi * 4 * (K + 8) + 4 * K + row
        negative_zp = tl.load(AInt + meta)
        activation_scale = tl.load(AFloat + meta + 16)
        corrected = values + negative_zp * weight_sum
        result = corrected.to(tl.float32) * (activation_scale * weight_scale)
        if BF16_OUTPUT == 2:
            result = result.to(tl.bfloat16).to(tl.uint16, bitcast=True)
        elif BF16_OUTPUT:
            result = _output_bf16_bits(result)
        if PAIR_N:
            offset = (
                (cols // PAIR_N) * M * PAIR_N + (mi * 16 + row) * PAIR_N + cols % PAIR_N
            )
        else:
            offset = (mi * 16 + row) * N + cols
        tl.store(C + offset, result, mi * 16 + row < M)


@triton.jit
def _wide_linear_kernel(
    A,
    AInt,
    AFloat,
    W,
    WSum,
    WScale,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
    TILE_N: tl.constexpr,
    ROW_EPILOGUE: tl.constexpr,
    BF16_OUTPUT: tl.constexpr = False,
    PAIR_N: tl.constexpr = 0,
    COLUMN_MAJOR: tl.constexpr = False,
):
    tiles_n = N // TILE_N
    begin = tle.program_id(0) * CHUNK
    end = tl.minimum(begin + CHUNK, tl.cdiv(M, 16) * tiles_n)
    lanes = tl.arange(0, 2048)
    rr, cc = tl.arange(0, 16), tl.arange(0, 16)
    for job in tl.range(begin, end, loop_unroll_factor=1):
        if COLUMN_MAJOR:
            mi, ni = job % tl.cdiv(M, 16), job // tl.cdiv(M, 16)
        else:
            mi, ni = job // tiles_n, job % tiles_n
        c0 = tl.zeros((16, 16), tl.int32)
        if TILE_N >= 32:
            c1 = tl.zeros((16, 16), tl.int32)
        if TILE_N == 64:
            c2 = tl.zeros((16, 16), tl.int32)
            c3 = tl.zeros((16, 16), tl.int32)
        for group in tl.range(0, K // 128, loop_unroll_factor=1):
            # Native LHS has 16 zero-points + 16 scales after each M16 panel.
            a = tl.load(A + mi * 16 * (K + 8) + group * 2048 + lanes).reshape(32, 16, 4)
            w0 = tl.load(
                W + (ni * (TILE_N // 16)) * 16 * K + group * 2048 + lanes
            ).reshape(32, 16, 4)
            if TILE_N >= 32:
                w1 = tl.load(
                    W + (ni * (TILE_N // 16) + 1) * 16 * K + group * 2048 + lanes
                ).reshape(32, 16, 4)
            if TILE_N == 64:
                w2 = tl.load(
                    W + (ni * (TILE_N // 16) + 2) * 16 * K + group * 2048 + lanes
                ).reshape(32, 16, 4)
            if TILE_N == 64:
                w3 = tl.load(
                    W + (ni * (TILE_N // 16) + 3) * 16 * K + group * 2048 + lanes
                ).reshape(32, 16, 4)
            a = a.permute(1, 0, 2).reshape(16, 128)
            w0 = w0.permute(0, 2, 1).reshape(128, 16)
            if TILE_N >= 32:
                w1 = w1.permute(0, 2, 1).reshape(128, 16)
            if TILE_N == 64:
                w2 = w2.permute(0, 2, 1).reshape(128, 16)
            if TILE_N == 64:
                w3 = w3.permute(0, 2, 1).reshape(128, 16)
            c0 += tl.dot(a, w0, out_dtype=tl.int32)
            if TILE_N >= 32:
                c1 += tl.dot(a, w1, out_dtype=tl.int32)
            if TILE_N == 64:
                c2 += tl.dot(a, w2, out_dtype=tl.int32)
            if TILE_N == 64:
                c3 += tl.dot(a, w3, out_dtype=tl.int32)
        meta = mi * 4 * (K + 8) + 4 * K + rr
        negative_zp = tl.load(AInt + meta)
        activation_scale = tl.load(AFloat + meta + 16)
        rows = mi * 16 + rr
        cols = ni * TILE_N + cc
        if ROW_EPILOGUE:
            _store_scaled_rows(
                C,
                c0,
                AInt,
                AFloat,
                WSum,
                WScale,
                mi,
                cols,
                M,
                N,
                K,
                BF16_OUTPUT,
                PAIR_N,
            )
            if TILE_N >= 32:
                _store_scaled_rows(
                    C,
                    c1,
                    AInt,
                    AFloat,
                    WSum,
                    WScale,
                    mi,
                    cols + 16,
                    M,
                    N,
                    K,
                    BF16_OUTPUT,
                    PAIR_N,
                )
            if TILE_N == 64:
                _store_scaled_rows(
                    C,
                    c2,
                    AInt,
                    AFloat,
                    WSum,
                    WScale,
                    mi,
                    cols + 32,
                    M,
                    N,
                    K,
                    BF16_OUTPUT,
                    PAIR_N,
                )
                _store_scaled_rows(
                    C,
                    c3,
                    AInt,
                    AFloat,
                    WSum,
                    WScale,
                    mi,
                    cols + 48,
                    M,
                    N,
                    K,
                    BF16_OUTPUT,
                    PAIR_N,
                )
        else:
            _store_scaled(
                C,
                c0,
                negative_zp,
                activation_scale,
                WSum,
                WScale,
                rows,
                cols,
                M,
                N,
                BF16_OUTPUT,
            )
            if TILE_N >= 32:
                _store_scaled(
                    C,
                    c1,
                    negative_zp,
                    activation_scale,
                    WSum,
                    WScale,
                    rows,
                    cols + 16,
                    M,
                    N,
                    BF16_OUTPUT,
                )
            if TILE_N == 64:
                _store_scaled(
                    C,
                    c2,
                    negative_zp,
                    activation_scale,
                    WSum,
                    WScale,
                    rows,
                    cols + 32,
                    M,
                    N,
                    BF16_OUTPUT,
                )
                _store_scaled(
                    C,
                    c3,
                    negative_zp,
                    activation_scale,
                    WSum,
                    WScale,
                    rows,
                    cols + 48,
                    M,
                    N,
                    BF16_OUTPUT,
                )


def _validate(packed, weight, sums, scales, m, n, k, workers, chunk, tile_n):
    if (
        m < 0
        or n <= 0
        or n % 64
        or k < 256
        or k > 32768
        or k % 128
        or workers < 1
        or chunk not in (1, 2, 4, 8, 16, 32, 64, 128)
        or tile_n not in (16, 32, 64)
    ):
        raise ValueError("Invalid W8A8 packed dimensions or launch configuration")
    specs = [
        (packed, torch.uint8, ((m + 15) // 16 * 16 * (k + 8),)),
        (weight, torch.int8, (n // 16, k // 128, 32, 16, 4)),
        (sums, torch.int32, (n,)),
        (scales, torch.float32, (n,)),
    ]
    for tensor, dtype, shape in specs:
        if (
            tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError("Invalid W8A8 packed tensor dtype, shape or strides")


def _run(
    packed,
    weight,
    sums,
    scales,
    m,
    n,
    k,
    workers=18,
    chunk=16,
    tile_n=16,
    row_epilogue=True,
):
    _validate(packed, weight, sums, scales, m, n, k, workers, chunk, tile_n)
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (packed, weight, sums, scales)
    ):
        raise RuntimeError("Packed W8A8 SME2 is inference-only")
    if any(t.device.type != "cpu" for t in (packed, weight, sums, scales)):
        raise ValueError("W8A8 packed SME2 requires CPU tensors")
    # CPU backend checks the selected host target; callers must also validate
    # the native packing ABI (512-bit SVL) before producing these buffers.
    out = torch.empty((m, n), dtype=torch.float32, device=packed.device)
    if m:
        jobs = triton.cdiv(m, 16) * (n // tile_n)
        _wide_linear_kernel[(triton.cdiv(jobs, chunk),)](
            packed.view(torch.int8),
            packed.view(torch.int32),
            packed.view(torch.float32),
            weight,
            sums,
            scales,
            out,
            m,
            n,
            k,
            chunk,
            tile_n,
            row_epilogue,
            num_cpu_threads=workers,
            enable_sme2=True,
            sme_vector_bits=512,
            enable_fast_math=False,
            enable_fp_fusion=False,
        )
    return out


_lib = torch.library.Library("flag_gems_arm", "FRAGMENT")
_lib.define(
    "w8a8_packed_sme2(Tensor packed, Tensor weight, Tensor sums, Tensor scales, int m, int n, int k, "
    "int workers=18, int chunk=16, int tile_n=16, bool row_epilogue=True) -> Tensor"
)
_lib.impl("w8a8_packed_sme2", _run, "CPU")


@torch.library.register_fake("flag_gems_arm::w8a8_packed_sme2")
def _fake(
    packed,
    weight,
    sums,
    scales,
    m,
    n,
    k,
    workers=18,
    chunk=16,
    tile_n=16,
    row_epilogue=True,
):
    _validate(packed, weight, sums, scales, m, n, k, workers, chunk, tile_n)
    return packed.new_empty((m, n), dtype=torch.float32)


def w8a8_packed_sme2(
    packed,
    weight,
    sums,
    scales,
    m,
    n,
    k,
    *,
    workers=18,
    chunk=16,
    tile_n=16,
    row_epilogue=True,
):
    """Inference-only; returns FP32. Bias/BF16 conversion stay caller-owned."""
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (packed, weight, sums, scales)
    ):
        raise RuntimeError("Packed W8A8 SME2 is inference-only")
    return torch.ops.flag_gems_arm.w8a8_packed_sme2(
        packed, weight, sums, scales, m, n, k, workers, chunk, tile_n, row_epilogue
    )


def w8a8_packed_sme2_bf16(
    packed,
    weight,
    sums,
    scales,
    m,
    n,
    k,
    *,
    workers=18,
    chunk=16,
    tile_n=16,
    pair_n=0,
    hardware_bf16=False,
    column_major=False,
):
    """Experimental bias-free direct BF16 output; canonical NaNs and RNE."""
    _validate(packed, weight, sums, scales, m, n, k, workers, chunk, tile_n)
    if pair_n and (pair_n * 2 != n or pair_n % 64):
        raise ValueError("Invalid paired projection dimensions")
    tensors = (packed, weight, sums, scales)
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("Expected CPU tensors")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise ValueError("Inference only")
    out = torch.empty((m, n), dtype=torch.int16)
    if m:
        jobs = triton.cdiv(m, 16) * (n // tile_n)
        _wide_linear_kernel[(triton.cdiv(jobs, chunk),)](
            packed.view(torch.int8),
            packed.view(torch.int32),
            packed.view(torch.float32),
            weight,
            sums,
            scales,
            out,
            m,
            n,
            k,
            chunk,
            tile_n,
            True,
            2 if hardware_bf16 else 1,
            pair_n,
            column_major,
            num_cpu_threads=workers,
            enable_sme2=True,
            sme_vector_bits=512,
            enable_fast_math=False,
            enable_fp_fusion=False,
        )
    result = out.view(torch.bfloat16)
    return result.reshape(2, m, pair_n) if pair_n else result
