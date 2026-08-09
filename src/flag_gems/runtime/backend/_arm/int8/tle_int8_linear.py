"""Drop-in replacement for nn.Linear using Triton SDOT decode + I8MM prefill.

Decode (M=1, BF16): one ordinary Triton kernel quantizes the activation, then
another ordinary ``tl.dot`` kernel consumes an SDOT-ready N-blocked layout.
The CPU compiler recognizes the load/reshape/transpose/dot graph and emits
AArch64 SDOT directly; no TLE frontend op or external GEMV is required.

Prefill (M>1): one ordinary Triton kernel quantizes directly into KAI's
M4/K8 activation ABI, then an ordinary ``tl.dot`` kernel consumes KAI N4/K8
weight panels. Target-aware CPU lowering emits SVE2 i8mm ``smmla`` without an
external matmul call. The same path remains faster than the former row-major
kernel through the measured long-prefill shapes, so the default cap covers
all practical sequence lengths.

Unsupported/capped shapes use the compatibility torch._int_mm path. Its
row-major weight is reconstructed lazily rather than retained as a third
full INT8 copy during normal inference.

The class also exposes the blocked packs and padded scales consumed by
FusedMLPWrapper, so gate/up can share quantization and a joined generated
matrix stage.
"""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:
    from triton.language.extra.cpu.tle_ops import (
        sdot_gemv_fused_bf16 as _cpu_fused_gemv,
        sdot_gemv_prequant as _cpu_prequant_gemv,
        sdot_gemv_whole as _cpu_whole_gemv,
    )

    _TLE_DECODE_AVAILABLE = True
except ImportError:
    # Official Triton-CPU 3.7 does not expose the development tree's TLE
    # frontend module. Keep all ordinary Q8 decode and prefill kernels
    # importable: production uses native tl.dot -> SDOT/I8MM JIT when an
    # optional generated AOT bundle is absent.
    _cpu_fused_gemv = None
    _cpu_prequant_gemv = None
    _cpu_whole_gemv = None
    _TLE_DECODE_AVAILABLE = False

from .aot_w8_backend import create_aot_w8_backend
from ..profile_range import profile_range

# Supported prefill shapes use the packed KAI-layout ordinary-Triton kernels
# below and never require a process-global aten::_int_mm override.  The final
# compatibility path intentionally uses the current ATen implementation unless
# a shape-owning caller explicitly opts into FlagGems' generic override.


@triton.jit
def _tle_fused_bf16_gemv_kernel(
    x_ptr,
    b_packed_ptr,
    w_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Codegen GEMV: BF16 x @ packed INT8 W → one BF16 output block.

    cpu::SdotGemvFusedBf16Op is expanded by the compiler into quantization,
    packed loads, SDOT accumulation and dequantization. The Triton grid owns
    parallelism across N; no external GEMV symbol is called.
    """
    n_start = tl.program_id(0) * BLOCK_N
    _cpu_fused_gemv(
        x_ptr,
        b_packed_ptr,
        w_scale_ptr,
        out_ptr,
        K,
        N,
        n_start,
        BLOCK_N,
    )


@triton.jit
def _tle_whole_bf16_gemv_kernel(
    x_ptr,
    b_packed_ptr,
    w_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Codegen a complete BF16 W8 projection in fixed SDOT microtiles.

    Unlike the per-block kernel above, activation quantization happens once.
    The compiler emits a rolled output loop whose TILE_N/4 accumulators fit
    the AArch64 vector register file.
    """
    _cpu_whole_gemv(
        x_ptr,
        b_packed_ptr,
        w_scale_ptr,
        out_ptr,
        K,
        N,
        TILE_N,
    )


@triton.jit
def _tle_prequant_bf16_gemv_kernel(
    x_q_ptr,
    x_scale_ptr,
    b_packed_ptr,
    w_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reuse one quantized activation across spill-free SDOT grid blocks."""
    n_start = tl.program_id(0) * BLOCK_N
    _cpu_prequant_gemv(
        x_q_ptr,
        x_scale_ptr,
        b_packed_ptr,
        w_scale_ptr,
        out_ptr,
        K,
        N,
        n_start,
        BLOCK_N,
    )


@triton.jit
def _quantize_bf16_w8_rne_kernel(
    x_ptr,
    q_ptr,
    x_scale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode quantizer matching the TLE whole-kernel RNE contract."""
    lanes8 = tl.arange(0, 8)
    absmax = tl.zeros((1,), dtype=tl.float32)
    full_k: tl.constexpr = (K // 32) * 32
    for off in tl.range(0, full_k, 32, loop_unroll_factor=1):
        values0 = tl.load(x_ptr + off + lanes8)
        values1 = tl.load(x_ptr + off + 8 + lanes8)
        values2 = tl.load(x_ptr + off + 16 + lanes8)
        values3 = tl.load(x_ptr + off + 24 + lanes8)
        bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(
            tl.int16
        )
        max01 = tl.where(bits0 > bits1, bits0, bits1)
        max23 = tl.where(bits2 > bits3, bits2, bits3)
        lane_max = tl.where(max01 > max23, max01, max23)
        block_bits = tl.max(lane_max, axis=0).to(tl.uint16)
        block_absmax = (
            block_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        )
        absmax = tl.maximum(absmax, block_absmax)
    if K % 32:
        tail_lanes = tl.arange(0, 32)
        tail_cols = full_k + tail_lanes
        tail_values = tl.load(
            x_ptr + tail_cols, mask=tail_cols < K, other=0.0
        ).to(
            tl.float32
        )
        absmax = tl.maximum(
            absmax, tl.max(tl.abs(tail_values), axis=0)
        )
    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 127.0 / absmax
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        values = tl.load(x_ptr + cols, mask=cols < K, other=0.0).to(
            tl.float32
        )
        # Match SdotGemvWholeOpLowering exactly: multiply by 127/absmax,
        # then round-to-nearest-even.  Reassociating this as x/scale changes
        # a few integer-boundary values and makes thread-count routing visible.
        scaled = values * inv_scale
        quantized = libdevice.rint(scaled).to(tl.int8)
        tl.store(q_ptr + cols, quantized, mask=cols < K)


@triton.jit
def _quantize_bf16_w8_vllm_trunc_kernel(
    x_ptr,
    q_ptr,
    x_scale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Match vLLM CPU dynamic W8 quantization on AArch64.

    vLLM's oneDNN front-end uses ``vcvtq_s32_f32`` after scaling.  That ACLE
    conversion truncates toward zero; it is not the RNE conversion used by
    the existing FlagGems W8 contract.  Keep this as a separate entry point
    so replacing vLLM preserves its numerical semantics without silently
    changing other FlagGems users.
    """
    lanes8 = tl.arange(0, 8)
    absmax = tl.zeros((1,), dtype=tl.float32)
    full_k: tl.constexpr = (K // 32) * 32
    for off in tl.range(0, full_k, 32, loop_unroll_factor=1):
        values0 = tl.load(x_ptr + off + lanes8)
        values1 = tl.load(x_ptr + off + 8 + lanes8)
        values2 = tl.load(x_ptr + off + 16 + lanes8)
        values3 = tl.load(x_ptr + off + 24 + lanes8)
        bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        max01 = tl.where(bits0 > bits1, bits0, bits1)
        max23 = tl.where(bits2 > bits3, bits2, bits3)
        lane_max = tl.where(max01 > max23, max01, max23)
        block_bits = tl.max(lane_max, axis=0).to(tl.uint16)
        block_absmax = block_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        absmax = tl.maximum(absmax, block_absmax)
    if K % 32:
        tail_lanes = tl.arange(0, 32)
        tail_cols = full_k + tail_lanes
        tail_values = tl.load(
            x_ptr + tail_cols, mask=tail_cols < K, other=0.0
        ).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(tail_values), axis=0))
    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 1.0 / scale
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        values = tl.load(
            x_ptr + cols, mask=cols < K, other=0.0
        ).to(tl.float32)
        scaled = tl.minimum(
            tl.maximum(values * inv_scale, -128.0), 127.0
        )
        tl.store(q_ptr + cols, scaled.to(tl.int8), mask=cols < K)


@triton.jit(do_not_specialize=["eps"])
def _rmsnorm_quantize_bf16_w8_rne_kernel(
    x_ptr,
    weight_ptr,
    q_ptr,
    x_scale_ptr,
    eps,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compiler-visible Llama RMSNorm followed by exact W8 RNE packing."""
    lanes = tl.arange(0, BLOCK_K)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        values = tl.load(x_ptr + off + lanes).to(tl.float32)
        sum_sq += tl.sum(values * values, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / K + eps)

    absmax = tl.zeros((1,), dtype=tl.float32)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        values = tl.load(x_ptr + off + lanes).to(tl.float32)
        weights = tl.load(weight_ptr + off + lanes).to(tl.float32)
        normalized = (values * rrms).to(tl.bfloat16).to(tl.float32)
        normalized = (normalized * weights).to(tl.bfloat16)
        bits = (normalized.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        block_bits = tl.max(bits, axis=0).to(tl.uint16)
        block_absmax = block_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        absmax = tl.maximum(absmax, block_absmax)
    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 127.0 / absmax
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)

    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        values = tl.load(x_ptr + off + lanes).to(tl.float32)
        weights = tl.load(weight_ptr + off + lanes).to(tl.float32)
        normalized = (values * rrms).to(tl.bfloat16).to(tl.float32)
        normalized = (normalized * weights).to(tl.bfloat16).to(tl.float32)
        quantized = libdevice.rint(normalized * inv_scale).to(tl.int8)
        tl.store(q_ptr + off + lanes, quantized)


@triton.jit(do_not_specialize=["eps"])
def _add_rmsnorm_quantize_bf16_w8_rne_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    updated_residual_ptr,
    q_ptr,
    x_scale_ptr,
    eps,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Residual add + Llama RMSNorm + exact W8 RNE packing."""
    lanes = tl.arange(0, BLOCK_K)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        summed = (
            tl.load(x_ptr + off + lanes).to(tl.float32)
            + tl.load(residual_ptr + off + lanes).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(updated_residual_ptr + off + lanes, summed)
        values = summed.to(tl.float32)
        sum_sq += tl.sum(values * values, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / K + eps)

    absmax = tl.zeros((1,), dtype=tl.float32)
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        values = tl.load(updated_residual_ptr + off + lanes).to(tl.float32)
        weights = tl.load(weight_ptr + off + lanes).to(tl.float32)
        normalized = (values * rrms).to(tl.bfloat16).to(tl.float32)
        normalized = (normalized * weights).to(tl.bfloat16)
        bits = (normalized.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
        block_bits = tl.max(bits, axis=0).to(tl.uint16)
        block_absmax = block_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        absmax = tl.maximum(absmax, block_absmax)
    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 127.0 / absmax
    tl.store(x_scale_ptr + tl.arange(0, 1), scale)

    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        values = tl.load(updated_residual_ptr + off + lanes).to(tl.float32)
        weights = tl.load(weight_ptr + off + lanes).to(tl.float32)
        normalized = (values * rrms).to(tl.bfloat16).to(tl.float32)
        normalized = (normalized * weights).to(tl.bfloat16).to(tl.float32)
        quantized = libdevice.rint(normalized * inv_scale).to(tl.int8)
        tl.store(q_ptr + off + lanes, quantized)


@triton.jit
def _w8_decode_sdot_kernel(
    x_q_ptr,
    x_scale_ptr,
    packed_ptr,
    weight_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    WHOLE_PROJECTION: tl.constexpr,
):
    """Ordinary Triton M1 W8 dot over an SDOT-ready blocked layout.

    The physical weight tile is ``[N lane, K lane]``.  Expressing its logical
    transpose with Triton reshape/transpose operations lets the CPU dot pass
    select four-lane Arm SDOT while retaining a rolled K loop.  A single
    program can stream the whole projection, or one program per output block
    can expose N-parallel work without changing the generated microkernel.
    """
    k_groups: tl.constexpr = K // 4
    cols = tl.arange(0, BLOCK_N)
    if WHOLE_PROJECTION:
        block_begin = 0
        block_end: tl.constexpr = N // BLOCK_N
    else:
        block_begin = tl.program_id(0)
        block_end = block_begin + 1

    for block in range(block_begin, block_end):
        accumulator = tl.zeros((1, BLOCK_N), dtype=tl.int32)
        for group in tl.range(
            0, k_groups, loop_unroll_factor=UNROLL
        ):
            packed_flat = tl.load(
                packed_ptr
                + (block * k_groups + group) * BLOCK_N * 4
                + tl.arange(0, BLOCK_N * 4)
            )
            weight = tl.trans(packed_flat.reshape((BLOCK_N, 4)))
            activation = tl.load(
                x_q_ptr + group * 4 + tl.arange(0, 4)
            ).reshape((1, 4))
            accumulator += tl.dot(
                activation, weight, out_dtype=tl.int32
            )

        activation_scale = tl.load(x_scale_ptr)
        weight_scale = tl.load(
            weight_scale_ptr + block * BLOCK_N + cols
        )
        result = (
            accumulator.to(tl.float32)
            * activation_scale
            * weight_scale[None, :]
        )
        tl.store(
            out_ptr + block * BLOCK_N + cols,
            result.reshape((BLOCK_N,)).to(tl.bfloat16),
        )


@triton.jit
def _quantize_rows_bf16_kernel(
    x_ptr,
    q_ptr,
    x_scale_ptr,
    M,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_qm,
    stride_qk,
    BLOCK_K: tl.constexpr,
):
    """BF16 -> per-row dynamic INT8 in one Triton launch.

    The old prefill path decomposed this into float/abs/amax/div/clamp/to ATen
    calls for every Linear.  Keeping both passes in one kernel removes those
    dispatcher calls and avoids FP32 intermediate tensors.
    """
    row = tl.program_id(0)
    row_ok = row < M
    absmax = tl.zeros((1,), dtype=tl.float32)
    # Keep K as a real rolled loop.  Static Python-range expansion at K=1024
    # produced a 4K-line object and hundreds of spill/reload references.
    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        mask = row_ok & (cols < K)
        x = tl.load(
            x_ptr + row * stride_xm + cols * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x), axis=0))

    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    tl.store(x_scale_ptr + row + tl.arange(0, 1), scale)

    for off in tl.range(0, K, BLOCK_K, loop_unroll_factor=1):
        cols = off + tl.arange(0, BLOCK_K)
        mask = cols < K
        x = tl.load(
            x_ptr + row * stride_xm + cols * stride_xk,
            mask=row_ok & mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.minimum(tl.maximum(x / scale, -128.0), 127.0).to(tl.int8)
        tl.store(q_ptr + row * stride_qm + cols * stride_qk, q, mask=mask)


@triton.jit
def _int8mm_dequant_bf16_kernel(
    a_ptr,
    b_ptr,
    x_scale_ptr,
    w_scale_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT8 tl.dot + per-row/per-channel dequant + BF16 store.

    ``tl.dot`` is intentionally visible in this Triton kernel so the CPU
    lowering selects the SVE2 i8mm path for prefill tiles.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for off in range(0, K, BLOCK_K):
        offs_k = off + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
        acc += tl.dot(a, b)

    x_scale = tl.load(x_scale_ptr + offs_m)
    w_scale = tl.load(w_scale_ptr + offs_n)
    out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(tl.bfloat16),
    )


@triton.jit
def _pack_lhs_w8_i8mm_kai_kernel(
    x_ptr,
    lhs_packed_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
    FULL_ROWS: tl.constexpr = False,
):
    """Quantize one BF16 row directly into the KAI M4/K8 physical ABI."""
    row = tl.program_id(0)
    if FULL_ROWS:
        row_ok = True
        source_row = row
    else:
        row_ok = row < M
        source_row = tl.minimum(row, M - 1)
    panel = row // 4
    panel_row = row % 4
    panel_stride: tl.constexpr = 4 * K + 16
    row_base = x_ptr + source_row * stride_xm

    if FULL_ROWS:
        # Finite positive BF16 encodings are monotonic. Combine four K8
        # slices lane-wise before the horizontal reduction, cutting the
        # reduction count in half without widening the live value set.
        lanes8 = tl.arange(0, 8)
        absmax = tl.zeros((1,), dtype=tl.float32)
        for off in tl.range(0, K, 32, loop_unroll_factor=1):
            values0 = tl.load(row_base + off + lanes8)
            values1 = tl.load(row_base + off + 8 + lanes8)
            values2 = tl.load(row_base + off + 16 + lanes8)
            values3 = tl.load(row_base + off + 24 + lanes8)
            bits0 = (
                values0.to(tl.uint16, bitcast=True) & 0x7FFF
            ).to(tl.int16)
            bits1 = (
                values1.to(tl.uint16, bitcast=True) & 0x7FFF
            ).to(tl.int16)
            bits2 = (
                values2.to(tl.uint16, bitcast=True) & 0x7FFF
            ).to(tl.int16)
            bits3 = (
                values3.to(tl.uint16, bitcast=True) & 0x7FFF
            ).to(tl.int16)
            max01 = tl.where(bits0 > bits1, bits0, bits1)
            max23 = tl.where(bits2 > bits3, bits2, bits3)
            lane_max = tl.where(max01 > max23, max01, max23)
            block_absmax_bits = tl.max(lane_max, axis=0).to(tl.uint16)
            block_absmax = (
                block_absmax_bits.to(tl.bfloat16, bitcast=True).to(
                    tl.float32
                )
            )
            absmax = tl.maximum(absmax, block_absmax)
    else:
        lanes16 = tl.arange(0, 16)
        absmax = tl.zeros((1,), dtype=tl.float32)
        for off in tl.range(0, K, 16, loop_unroll_factor=1):
            values = tl.load(row_base + off + lanes16).to(tl.float32)
            values = tl.where(row_ok, values, 0.0)
            absmax = tl.maximum(absmax, tl.max(tl.abs(values), axis=0))

    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 127.0 / absmax
    lanes8 = tl.arange(0, 8)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(row_base + off + lanes8).to(tl.float32)
        if not FULL_ROWS:
            values = tl.where(row_ok, values, 0.0)
        # Match the decode quantizer and compressed-tensors INT semantics:
        # multiply by 127/absmax and round-to-nearest-even.  The old prefill
        # route truncated here, so M=1 and M>1 represented the same token
        # differently while its tailored test reference hid the mismatch.
        scaled = values * inv_scale
        if not FULL_ROWS:
            scaled = tl.minimum(tl.maximum(scaled, -128.0), 127.0)
        quantized = libdevice.rint(scaled).to(tl.int8)
        packed_offset = (
            panel * panel_stride
            + (off // 8) * 32
            + panel_row * 8
        )
        tl.store(lhs_packed_ptr + packed_offset + lanes8, quantized)

    scale_ptr = (lhs_packed_ptr + panel * panel_stride + 4 * K).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(scale_ptr + panel_row + tl.arange(0, 1), scale)


@triton.jit
def _pack_lhs_w8_i8mm_kai_vllm_trunc_kernel(
    x_ptr,
    lhs_packed_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
    FULL_ROWS: tl.constexpr = False,
):
    """KAI M4/K8 LHS pack with vLLM/ACLE truncation semantics."""
    row = tl.program_id(0)
    if FULL_ROWS:
        row_ok = True
        source_row = row
    else:
        row_ok = row < M
        source_row = tl.minimum(row, M - 1)
    panel = row // 4
    panel_row = row % 4
    panel_stride: tl.constexpr = 4 * K + 16
    row_base = x_ptr + source_row * stride_xm

    if FULL_ROWS:
        lanes8 = tl.arange(0, 8)
        absmax = tl.zeros((1,), dtype=tl.float32)
        for off in tl.range(0, K, 32, loop_unroll_factor=1):
            values0 = tl.load(row_base + off + lanes8)
            values1 = tl.load(row_base + off + 8 + lanes8)
            values2 = tl.load(row_base + off + 16 + lanes8)
            values3 = tl.load(row_base + off + 24 + lanes8)
            bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(
                tl.int16
            )
            bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(
                tl.int16
            )
            bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(
                tl.int16
            )
            bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(
                tl.int16
            )
            max01 = tl.where(bits0 > bits1, bits0, bits1)
            max23 = tl.where(bits2 > bits3, bits2, bits3)
            lane_max = tl.where(max01 > max23, max01, max23)
            block_bits = tl.max(lane_max, axis=0).to(tl.uint16)
            block_absmax = block_bits.to(
                tl.bfloat16, bitcast=True
            ).to(tl.float32)
            absmax = tl.maximum(absmax, block_absmax)
    else:
        lanes16 = tl.arange(0, 16)
        absmax = tl.zeros((1,), dtype=tl.float32)
        for off in tl.range(0, K, 16, loop_unroll_factor=1):
            values = tl.load(row_base + off + lanes16).to(tl.float32)
            values = tl.where(row_ok, values, 0.0)
            absmax = tl.maximum(absmax, tl.max(tl.abs(values), axis=0))

    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 1.0 / scale
    lanes8 = tl.arange(0, 8)
    for off in tl.range(0, K, 8, loop_unroll_factor=1):
        values = tl.load(row_base + off + lanes8).to(tl.float32)
        if not FULL_ROWS:
            values = tl.where(row_ok, values, 0.0)
        scaled = tl.minimum(
            tl.maximum(values * inv_scale, -128.0), 127.0
        )
        packed_offset = (
            panel * panel_stride + (off // 8) * 32 + panel_row * 8
        )
        tl.store(
            lhs_packed_ptr + packed_offset + lanes8,
            scaled.to(tl.int8),
        )

    scale_ptr = (lhs_packed_ptr + panel * panel_stride + 4 * K).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(scale_ptr + panel_row + tl.arange(0, 1), scale)


@triton.jit
def _w8_prefill_i8mm_kai_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Ordinary tl.dot over KAI M16/N4 K8 panels, lowered to target I8MM."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    panel_stride: tl.constexpr = 4 * K + 16
    rhs_panel_stride: tl.constexpr = 4 * K + 16
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 4)
    panel_bytes = tl.arange(0, 128)
    accumulator = tl.zeros((BLOCK_M, 4), tl.int32)

    for chunk in range(0, K // 32):
        lhs_base = pid_m * 4 * panel_stride + chunk * 128
        rhs_base = pid_n * rhs_panel_stride + chunk * 128
        rhs = tl.load(
            rhs_packed_ptr + rhs_base + panel_bytes
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        lhs0 = tl.load(
            lhs_packed_ptr + lhs_base + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs1 = tl.load(
            lhs_packed_ptr + lhs_base + panel_stride + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs2 = tl.load(
            lhs_packed_ptr + lhs_base + 2 * panel_stride + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs3 = tl.load(
            lhs_packed_ptr + lhs_base + 3 * panel_stride + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs01 = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        lhs23 = tl.join(lhs2, lhs3).permute(2, 0, 1).reshape((8, 32))
        lhs = tl.join(lhs01, lhs23).permute(2, 0, 1).reshape((16, 32))
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)

    meta_lanes = tl.arange(0, 4)
    lhs_meta = pid_m * 4 * panel_stride + 4 * K
    rhs_meta = pid_n * rhs_panel_stride + 4 * K
    rhs_scale = tl.load(
        (rhs_packed_ptr + rhs_meta).to(tl.pointer_type(tl.float32)) + cols
    )
    lhs_scale0 = tl.load(
        (lhs_packed_ptr + lhs_meta).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    lhs_scale1 = tl.load(
        (lhs_packed_ptr + lhs_meta + panel_stride).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    lhs_scale2 = tl.load(
        (lhs_packed_ptr + lhs_meta + 2 * panel_stride).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    lhs_scale3 = tl.load(
        (lhs_packed_ptr + lhs_meta + 3 * panel_stride).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    lhs_scale01 = tl.join(lhs_scale0, lhs_scale1).permute(1, 0).reshape(8)
    lhs_scale23 = tl.join(lhs_scale2, lhs_scale3).permute(1, 0).reshape(8)
    lhs_scale = tl.join(lhs_scale01, lhs_scale23).permute(1, 0).reshape(16)
    result = (
        accumulator.to(tl.float32)
        * lhs_scale[:, None]
        * rhs_scale[None, :]
    )
    output_rows = pid_m * BLOCK_M + rows
    output_cols = pid_n * 4 + cols
    tl.store(
        out_ptr + output_rows[:, None] * N + output_cols[None, :],
        result.to(tl.bfloat16),
    )


@triton.jit
def _w8_prefill_i8mm_kai_short_tail_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute one M4 or M8 tail as one compiler-recognized packed dot."""
    pid_n = tl.program_id(1)
    panel_stride: tl.constexpr = 4 * K + 16
    rhs_panel_stride: tl.constexpr = 4 * K + 16
    cols = tl.arange(0, 4)
    panel_bytes = tl.arange(0, 128)
    accumulator = tl.zeros((BLOCK_M, 4), tl.int32)

    for chunk in range(0, K // 32):
        lhs_base = chunk * 128
        rhs_base = pid_n * rhs_panel_stride + chunk * 128
        rhs = tl.load(
            rhs_packed_ptr + rhs_base + panel_bytes
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        lhs0 = tl.load(
            lhs_packed_ptr + lhs_base + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        if BLOCK_M == 4:
            lhs = lhs0
        else:
            lhs1 = tl.load(
                lhs_packed_ptr + lhs_base + panel_stride + panel_bytes
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)

    meta_lanes = tl.arange(0, 4)
    rhs_scale = tl.load(
        (rhs_packed_ptr + pid_n * rhs_panel_stride + 4 * K).to(
            tl.pointer_type(tl.float32)
        )
        + cols
    )
    lhs_scale0 = tl.load(
        (lhs_packed_ptr + 4 * K).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    if BLOCK_M == 4:
        lhs_scale = lhs_scale0
    else:
        lhs_scale1 = tl.load(
            (lhs_packed_ptr + panel_stride + 4 * K).to(
                tl.pointer_type(tl.float32)
            )
            + meta_lanes
        )
        lhs_scale = tl.join(lhs_scale0, lhs_scale1).permute(
            1, 0
        ).reshape((8,))
    result = (
        accumulator.to(tl.float32)
        * lhs_scale[:, None]
        * rhs_scale[None, :]
    )
    rows = tl.arange(0, BLOCK_M)
    output_cols = pid_n * 4 + cols
    tl.store(
        out_ptr + rows[:, None] * N + output_cols[None, :],
        result.to(tl.bfloat16),
    )


@triton.jit
def _w8_prefill_i8mm_kai_m12_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """Compute a padded M12 tail as compiler-fused M8 and M4 dots."""
    pid_n = tl.program_id(1)
    panel_stride: tl.constexpr = 4 * K + 16
    rhs_panel_stride: tl.constexpr = 4 * K + 16
    cols = tl.arange(0, 4)
    panel_bytes = tl.arange(0, 128)
    accumulator8 = tl.zeros((8, 4), tl.int32)
    accumulator4 = tl.zeros((4, 4), tl.int32)

    for chunk in range(0, K // 32):
        lhs_base = chunk * 128
        rhs_base = pid_n * rhs_panel_stride + chunk * 128
        rhs = tl.load(
            rhs_packed_ptr + rhs_base + panel_bytes
        ).reshape((4, 4, 8)).permute(0, 2, 1).reshape((32, 4))
        lhs0 = tl.load(
            lhs_packed_ptr + lhs_base + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs1 = tl.load(
            lhs_packed_ptr + lhs_base + panel_stride + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs2 = tl.load(
            lhs_packed_ptr + lhs_base + 2 * panel_stride + panel_bytes
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs8 = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
        accumulator8 += tl.dot(lhs8, rhs, out_dtype=tl.int32)
        accumulator4 += tl.dot(lhs2, rhs, out_dtype=tl.int32)

    meta_lanes = tl.arange(0, 4)
    rhs_scale = tl.load(
        (rhs_packed_ptr + pid_n * rhs_panel_stride + 4 * K).to(
            tl.pointer_type(tl.float32)
        )
        + cols
    )
    scale0 = tl.load(
        (lhs_packed_ptr + 4 * K).to(tl.pointer_type(tl.float32))
        + meta_lanes
    )
    scale1 = tl.load(
        (lhs_packed_ptr + panel_stride + 4 * K).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    scale2 = tl.load(
        (lhs_packed_ptr + 2 * panel_stride + 4 * K).to(
            tl.pointer_type(tl.float32)
        )
        + meta_lanes
    )
    scale8 = tl.join(scale0, scale1).permute(1, 0).reshape((8,))
    result8 = (
        accumulator8.to(tl.float32)
        * scale8[:, None]
        * rhs_scale[None, :]
    )
    result4 = (
        accumulator4.to(tl.float32)
        * scale2[:, None]
        * rhs_scale[None, :]
    )
    output_cols = pid_n * 4 + cols
    tl.store(
        out_ptr + tl.arange(0, 8)[:, None] * N + output_cols[None, :],
        result8.to(tl.bfloat16),
    )
    tl.store(
        out_ptr + (8 + meta_lanes)[:, None] * N + output_cols[None, :],
        result4.to(tl.bfloat16),
    )


def linear_w8_vllm_dynamic(
    x: torch.Tensor,
    decode_rhs: torch.Tensor,
    prefill_rhs: torch.Tensor,
    weight_scale: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Ordinary-Triton mirror of the libtriton_jit vLLM W8 router.

    This is primarily a correctness and launch-overhead reference.  Production
    vLLM uses the C++ operator so the Python dispatcher is outside the hot path.
    """
    if x.dtype != torch.bfloat16 or x.shape[-1] != k:
        raise ValueError("vLLM W8 reference requires BF16 [..., K] input")
    shape = x.shape
    xf = x.reshape(-1, k).contiguous()
    m = xf.shape[0]
    if m == 1:
        decode_tile_n = select_w8_decode_tile_n(n, 64)
        x_q = torch.empty((k,), dtype=torch.int8)
        x_scale = torch.empty((1,), dtype=torch.float32)
        output = torch.empty((n,), dtype=torch.bfloat16)
        _quantize_bf16_w8_vllm_trunc_kernel[(1,)](
            xf,
            x_q,
            x_scale,
            K=k,
            BLOCK_K=16,
            num_warps=1,
            num_stages=1,
        )
        _w8_decode_sdot_kernel[(n // decode_tile_n,)](
            x_q,
            x_scale,
            decode_rhs,
            weight_scale,
            output,
            K=k,
            N=n,
            BLOCK_N=decode_tile_n,
            UNROLL=2,
            WHOLE_PROJECTION=False,
            num_warps=1,
            num_stages=1,
        )
        return output.reshape(*shape[:-1], n)

    m_kernel = 4 if m <= 4 else 8 if m <= 8 else 12 if m <= 12 else (
        (m + 15) // 16
    ) * 16
    panel_stride = 4 * k + 16
    lhs = torch.empty(
        ((m_kernel // 4) * panel_stride,), dtype=torch.int8
    )
    _pack_lhs_w8_i8mm_kai_vllm_trunc_kernel[(m_kernel,)](
        xf,
        lhs,
        m,
        xf.stride(0),
        K=k,
        FULL_ROWS=m == m_kernel,
        num_warps=1,
        num_stages=1,
    )
    output = torch.empty((m_kernel, n), dtype=torch.bfloat16)
    if m_kernel <= 8:
        _w8_prefill_i8mm_kai_short_tail_kernel[(1, n // 4)](
            lhs,
            prefill_rhs,
            output,
            N=n,
            K=k,
            BLOCK_M=m_kernel,
            num_warps=1,
            num_stages=1,
        )
    elif m_kernel == 12:
        _w8_prefill_i8mm_kai_m12_kernel[(1, n // 4)](
            lhs,
            prefill_rhs,
            output,
            N=n,
            K=k,
            num_warps=1,
            num_stages=1,
        )
    else:
        _w8_prefill_i8mm_kai_kernel[(m_kernel // 16, n // 4)](
            lhs,
            prefill_rhs,
            output,
            N=n,
            K=k,
            BLOCK_M=16,
            num_warps=1,
            num_stages=1,
        )
    return output[:m].reshape(*shape[:-1], n)


def _fused_prefill_enabled() -> bool:
    return os.getenv("FLAGGEMS_ARM_FUSED_PREFILL", "1").lower() in (
        "1",
        "true",
        "on",
    )


def _fused_prefill_max_m() -> int:
    """Largest flattened row count routed through the KAI-layout kernel."""
    try:
        return max(
            0,
            int(os.getenv("FLAGGEMS_ARM_FUSED_PREFILL_MAX_M", "1048576")),
        )
    except ValueError:
        return 1048576


def _fused_prefill_w8a8(
    x: torch.Tensor,
    weight_kn: torch.Tensor | None,
    weight_scale: torch.Tensor,
    weight_kai: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run the two-launch W8A8 prefill path, or return None if unsupported."""
    shape = x.shape
    K = shape[-1]
    M = x.numel() // K
    if weight_kn is not None:
        N = weight_kn.shape[1]
    elif weight_kai is not None:
        panel_stride = 4 * K + 16
        N = (weight_kai.numel() // panel_stride) * 4
    else:
        return None
    if (
        not _fused_prefill_enabled()
        or M <= 1
        or M > _fused_prefill_max_m()
        or K % 32 != 0
        or N % 4 != 0
    ):
        return None

    xf = x.reshape(M, K).contiguous()
    if weight_kai is not None:
        panel_stride = 4 * K + 16
        # M2-M12 is common in short-batch/speculative execution. Dedicated
        # M4/M8/M12 objects remove one to three quarters of the old padded M16
        # matrix work.  At M>=9, keep one M16-grid launch: splitting a main
        # tile and tail into two Python launches costs more than the saved
        # I8MM work on CIX.
        short_tail = M <= 12 and os.getenv(
            "FLAGGEMS_ARM_W8_SHORT_PREFILL", "1"
        ).lower() not in ("0", "false", "off")
        m_kernel = (
            (4 if M <= 4 else 8 if M <= 8 else 12)
            if short_tail
            else ((M + 15) // 16) * 16
        )
        lhs_packed = torch.empty(
            (m_kernel // 4) * panel_stride, dtype=torch.int8
        )
        # Shape-specialize the common exact M4/M8/M12/M16 cases.  Keeping one
        # row per program avoids the register pressure of a four-row
        # reduction, while removing the row predicate from both K loops.  A
        # non-multiple tail stays in the single-launch masked form below.
        if M == m_kernel:
            _pack_lhs_w8_i8mm_kai_kernel[(m_kernel,)](
                xf,
                lhs_packed,
                M,
                xf.stride(0),
                K=K,
                FULL_ROWS=True,
                num_warps=1,
                num_stages=1,
            )
        else:
            _pack_lhs_w8_i8mm_kai_kernel[(m_kernel,)](
                xf,
                lhs_packed,
                M,
                xf.stride(0),
                K=K,
                num_warps=1,
                num_stages=1,
            )
        out = torch.empty((m_kernel, N), dtype=torch.bfloat16)
        if m_kernel <= 8:
            _w8_prefill_i8mm_kai_short_tail_kernel[(1, N // 4)](
                lhs_packed,
                weight_kai,
                out,
                N=N,
                K=K,
                BLOCK_M=m_kernel,
                num_warps=1,
                num_stages=1,
            )
        elif short_tail:
            _w8_prefill_i8mm_kai_m12_kernel[(1, N // 4)](
                lhs_packed,
                weight_kai,
                out,
                N=N,
                K=K,
                num_warps=1,
                num_stages=1,
            )
        else:
            _w8_prefill_i8mm_kai_kernel[(m_kernel // 16, N // 4)](
                lhs_packed,
                weight_kai,
                out,
                N=N,
                K=K,
                BLOCK_M=16,
                num_warps=1,
                num_stages=1,
            )
        return out[:M].reshape(*shape[:-1], N)

    if N % 64 != 0 or weight_kn is None:
        return None
    if M == 2:
        block_m, block_k, m_kernel = 2, 4, M
    else:
        block_m, block_k = 8, 32
        m_kernel = ((M + block_m - 1) // block_m) * block_m

    q = torch.empty((m_kernel, K), dtype=torch.int8)
    x_scale = torch.empty((m_kernel,), dtype=torch.float32)
    _quantize_rows_bf16_kernel[(m_kernel,)](
        xf,
        q,
        x_scale,
        M,
        K,
        xf.stride(0),
        xf.stride(1),
        q.stride(0),
        q.stride(1),
        # K16 is the smallest BF16 masked-load shape selected reliably by
        # LLVM AArch64.  It matches K32 latency on CIX while shrinking the
        # quantizer object from 1671 to about 261 assembly lines.
        BLOCK_K=16,
    )

    # The quantizer already pads rows to ``m_kernel``.  Give the matmul a
    # matching output allocation so its hot store is unmasked; small BF16
    # masked stores are poorly selected by LLVM AArch64 and block N4 tiling.
    out = torch.empty((m_kernel, N), dtype=torch.bfloat16)
    _int8mm_dequant_bf16_kernel[(m_kernel // block_m, N // 64)](
        q,
        weight_kn,
        x_scale,
        weight_scale,
        out,
        M,
        N,
        K,
        q.stride(0),
        q.stride(1),
        weight_kn.stride(0),
        weight_kn.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_K=block_k,
    )
    return out[:M].reshape(*shape[:-1], N)


def pack_weights_sdot(w_kn: torch.Tensor) -> torch.Tensor:
    """Pack row-major [K, N] INT8 weight into SDOT-friendly [K//4, N//4, 4, 4].

    SDOT loads 4 consecutive K-bytes from one lane and broadcasts to 4 N-lanes.
    The packed layout ensures each SDOT tile is contiguous in memory for
    maximum L1 cache efficiency. Requires K%4==0 and N%4==0.
    """
    K, N = w_kn.shape
    if K % 4 != 0 or N % 4 != 0:
        raise ValueError(
            f"pack_weights_sdot requires K%4==0 and N%4==0, got K={K} N={N}"
        )
    return w_kn.reshape(K // 4, 4, N // 4, 4).permute(0, 2, 3, 1).contiguous()


def pack_weights_i8mm_kai(
    w_int8: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    """Pack [N,K] W8 weights/scales into KAI's contiguous N4/K8 ABI."""
    if w_int8.dtype != torch.int8 or w_int8.ndim != 2:
        raise TypeError("w_int8 must be a two-dimensional INT8 tensor")
    N, K = w_int8.shape
    if N % 4 or K % 32:
        raise ValueError(
            f"KAI W8 pack requires N%4=0 and K%32=0, got N={N} K={K}"
        )
    scale = w_scale.reshape(-1).to(torch.float32)
    if scale.numel() != N:
        raise ValueError(f"expected N={N} scales, got {scale.numel()}")
    panel_stride = 4 * K + 16
    packed = torch.empty((N // 4, panel_stride), dtype=torch.int8)
    values = (
        w_int8.reshape(N // 4, 4, K // 8, 8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(N // 4, 4 * K)
    )
    packed[:, : 4 * K].copy_(values)
    packed[:, 4 * K :].view(torch.float32).copy_(scale.reshape(N // 4, 4))
    return packed.reshape(-1)


def unpack_weights_i8mm_kai(
    packed: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    """Recover row-major [K,N] INT8 weights for a rare compatibility path."""
    panel_stride = 4 * k + 16
    if n % 4 or k % 32 or packed.numel() != (n // 4) * panel_stride:
        raise ValueError("packed KAI W8 tensor does not match N/K")
    values = packed.reshape(n // 4, panel_stride)[:, : 4 * k]
    weight_nk = (
        values.reshape(n // 4, k // 8, 4, 8)
        .permute(0, 2, 1, 3)
        .reshape(n, k)
    )
    return weight_nk.T.contiguous()


def pack_weights_sdot_blocked(
    packed_kmajor: torch.Tensor, block_n: int
) -> torch.Tensor:
    """Reorder SDOT tiles so one Triton N-block is a contiguous K stream."""
    k4, n4, ni, ki = packed_kmajor.shape
    groups = block_n // 4
    if block_n % 4 or n4 % groups:
        raise ValueError(
            f"blocked SDOT layout requires N divisible by block_n={block_n}"
        )
    return (
        packed_kmajor.reshape(k4, n4 // groups, groups, ni, ki)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def retile_weights_sdot_blocked(
    packed_blocked: torch.Tensor, block_n: int, tile_n: int
) -> torch.Tensor:
    """Split a block-major SDOT pack into smaller contiguous output tiles."""
    blocks, k4, groups, ni, ki = packed_blocked.shape
    if (
        block_n % tile_n
        or groups != block_n // 4
        or tile_n % 4
        or tile_n > block_n
    ):
        raise ValueError(
            f"cannot retile SDOT block {block_n} to tile {tile_n}"
        )
    tiles_per_block = block_n // tile_n
    micro_groups = tile_n // 4
    return (
        packed_blocked.reshape(
            blocks, k4, tiles_per_block, micro_groups, ni, ki
        )
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(blocks * tiles_per_block, k4, micro_groups, ni, ki)
        .contiguous()
    )


def select_w8_decode_tile_n(n_codegen: int, block_n: int) -> int:
    """Choose a spill-free SDOT tile shared by JIT and optional AOT.

    Large vocabulary projections are bandwidth-bound and run about 5% faster
    with 32 outputs per K stream on CIX. Decoder, QKV and MLP projections keep
    64 outputs to amortize loop/epilogue overhead.
    """
    tile_n = 32 if n_codegen >= 32768 else 64
    tile_n = min(tile_n, block_n)
    while block_n % tile_n:
        tile_n -= 4
    return tile_n


class TLEInt8Linear(torch.nn.Module):
    """nn.Linear replacement with SDOT decode + KAI-layout I8MM prefill.

    Args:
        w_int8:   [N, K] int8 tensor (pre-quantized weight, same layout as nn.Linear's
                  .weight.data attribute but dtype=int8).
        w_scale:  [N] fp32 tensor (per-column weight scales); scalar broadcasted
                  tensors also accepted.

    Required: K % 4 == 0 and N % 4 == 0 (SDOT lane requirement).

    Attributes exposed for downstream fusion passes (e.g. patch_qwen3_mlp):
        _packed_codegen: block-major INT8 pack for compiler-generated decode
        _packed_prefill_kai: N4/K8 INT8 pack for compiler-generated prefill
        _w_int8_kn: lazy [K,N] compatibility copy (normally None)
        _w_scale:  [N] fp32                 — per-column scale
        K, N:      ints
    """

    def __init__(self, w_int8: torch.Tensor, w_scale: torch.Tensor):
        super().__init__()
        if w_int8.dtype != torch.int8:
            raise TypeError(f"w_int8 must be int8, got {w_int8.dtype}")
        if w_int8.ndim != 2:
            raise ValueError(
                f"w_int8 must have [N, K] rank 2, got {w_int8.shape}"
            )
        self.N, self.K = w_int8.shape
        if self.K % 4 or self.N % 4:
            raise ValueError(
                "W8 SDOT decode requires K and N divisible by 4, "
                f"got K={self.K}, N={self.N}"
            )
        w_kn = w_int8.t().contiguous()  # [K, N]
        # Large compiler-visible blocks amortize the CPU launcher and make
        # each program's packed-weight K stream long enough for hardware
        # prefetch.  Padding also avoids falling back to tiny blocks for vocab
        # sizes such as 151936.
        # 512 outputs balance sequential weight streaming, generated code size
        # and reuse by the fused gate+up lowering (two accumulator banks).
        self._codegen_block_n = min(512, self.N)
        self._N_codegen = (
            (self.N + self._codegen_block_n - 1)
            // self._codegen_block_n
            * self._codegen_block_n
        )
        if self._N_codegen == self.N:
            w_kn_codegen = w_kn
        else:
            w_kn_codegen = torch.zeros(
                (self.K, self._N_codegen), dtype=torch.int8
            )
            w_kn_codegen[:, : self.N] = w_kn
        packed_codegen_kmajor = pack_weights_sdot(w_kn_codegen)
        self._packed_codegen = pack_weights_sdot_blocked(
            packed_codegen_kmajor, self._codegen_block_n
        )
        self._whole_codegen = False
        self._whole_tile_n = None
        self._ordinary_aot = None
        self._ordinary_aot_attempted = False
        # Do not retain the old K-major pack. It was needed by the external C
        # runtime, but compiler-generated decode and MLP both consume the
        # block-major pack. Keeping it duplicated every model W8 weight.
        if os.getenv("FLAGGEMS_ARM_KEEP_LEGACY_PACK", "0") == "1":
            self._packed_legacy = pack_weights_sdot(w_kn)
        self._w_int8_kn = w_kn  # released below when KAI prefill is available
        scale = w_scale.reshape(-1).to(torch.float32)
        if scale.numel() == 1:
            scale = scale.expand(self.N)
        elif scale.numel() != self.N:
            raise ValueError(
                f"w_scale must be scalar or have N={self.N} elements, "
                f"got {scale.numel()}"
            )
        self._w_scale = scale.contiguous()  # [N]
        if self._N_codegen == self.N:
            self._w_scale_codegen = self._w_scale
        else:
            self._w_scale_codegen = torch.zeros(
                self._N_codegen, dtype=torch.float32
            )
            self._w_scale_codegen[: self.N] = self._w_scale
        # Short-prefill ordinary tl.dot consumes the same N4/K8 panel ABI as
        # KleidiAI, but remains compiler generated.  Keep it separately from
        # the SDOT decode pack because the two kernels traverse K differently.
        self._packed_prefill_kai = None
        if self.K % 32 == 0 and self.N % 4 == 0:
            self._packed_prefill_kai = pack_weights_i8mm_kai(
                w_int8, self._w_scale
            )
            # Decode and KAI prefill now have their own physical packs.  Do
            # not retain a third full row-major weight copy.  It is recovered
            # lazily only if an explicit prefill cap forces compatibility.
            self._w_int8_kn = None

    def _get_w_int8_kn(self) -> torch.Tensor:
        if self._w_int8_kn is None:
            if self._packed_prefill_kai is None:
                raise RuntimeError("no W8 weight representation is available")
            self._w_int8_kn = unpack_weights_i8mm_kai(
                self._packed_prefill_kai, self.N, self.K
            )
        return self._w_int8_kn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        M = x.numel() // shape[-1]
        if M == 1 and x.dtype == torch.bfloat16:
            # Decode fast path — compiler-generated packed SDOT GEMV.
            xc = x if x.is_contiguous() else x.contiguous()
            exact_output = self._N_codegen == self.N
            out = torch.empty(
                (*shape[:-1], self.N)
                if exact_output
                else (self._N_codegen,),
                dtype=torch.bfloat16,
            )
            block_n = self._codegen_block_n
            if (
                torch.get_num_threads() == 1
                and not self._ordinary_aot_attempted
            ):
                # The vocabulary projection is bandwidth-bound and performs
                # best with fewer live accumulators than the smaller decoder
                # projections on Arm SVE2/SDOT.
                aot_block_n = select_w8_decode_tile_n(
                    self._N_codegen, block_n
                )
                self._ordinary_aot = create_aot_w8_backend(
                    self.K, self._N_codegen, aot_block_n
                )
                self._ordinary_aot_attempted = True
            if self._ordinary_aot is not None:
                if self._packed_codegen is None:
                    raise RuntimeError("ordinary W8 AOT pack has been released")
                if not self._whole_codegen:
                    self._packed_codegen = retile_weights_sdot_blocked(
                        self._packed_codegen,
                        block_n,
                        self._ordinary_aot.block_n,
                    )
                    self._whole_tile_n = self._ordinary_aot.block_n
                    self._whole_codegen = True
                elif self._whole_tile_n != self._ordinary_aot.block_n:
                    raise RuntimeError(
                        "ordinary W8 AOT pack has an incompatible tile size"
                    )
                with profile_range("triton::w8_decode_ordinary_aot"):
                    self._ordinary_aot(
                        xc,
                        self._packed_codegen,
                        self._w_scale_codegen,
                        out,
                    )
                if exact_output:
                    return out
                return out[: self.N].reshape(*shape[:-1], self.N)
            with profile_range("triton::w8_decode_sdot_codegen"):
                whole_mode = os.getenv(
                    "FLAGGEMS_ARM_W8_WHOLE_CODEGEN", "auto"
                ).lower()
                use_whole = whole_mode in {"1", "true", "on"} or (
                    whole_mode == "auto" and torch.get_num_threads() == 1
                )
                tile_n = select_w8_decode_tile_n(
                    self._N_codegen, block_n
                )
                if not self._whole_codegen:
                    self._packed_codegen = retile_weights_sdot_blocked(
                        self._packed_codegen, block_n, tile_n
                    )
                    self._whole_tile_n = tile_n
                    self._whole_codegen = True
                elif self._whole_tile_n != tile_n:
                    raise RuntimeError(
                        "live W8 Linear has an incompatible SDOT tile size"
                    )
                # Quantize once, then share the INT8 activation across either
                # one rolled whole-projection program or a selected N-tile
                # program grid (BN32 for vocab, BN64 for decoder shapes).
                # Both stages are ordinary Triton and the matrix stage exposes
                # its tl.dot graph directly to the CPU SDOT lowering.
                x_q = torch.empty((self.K,), dtype=torch.int8)
                x_scale = torch.empty((1,), dtype=torch.float32)
                _quantize_bf16_w8_rne_kernel[(1,)](
                    xc,
                    x_q,
                    x_scale,
                    K=self.K,
                    BLOCK_K=16,
                )
                grid = (1,) if use_whole else (
                    self._N_codegen // tile_n,
                )
                _w8_decode_sdot_kernel[grid](
                    x_q,
                    x_scale,
                    self._packed_codegen,
                    self._w_scale_codegen,
                    out,
                    K=self.K,
                    N=self._N_codegen,
                    BLOCK_N=tile_n,
                    UNROLL=2,
                    WHOLE_PROJECTION=use_whole,
                )
            if exact_output:
                return out
            return out[: self.N].reshape(*shape[:-1], self.N)

        fused = _fused_prefill_w8a8(
            x,
            self._w_int8_kn,
            self._w_scale,
            self._packed_prefill_kai,
        )
        if fused is not None:
            return fused

        # Compatibility path: decomposed per-row quant -> _int_mm -> dequant.
        weight_kn = self._get_w_int8_kn()
        xf = x.reshape(-1, self.K).contiguous()
        xf32 = xf.float()
        absmax = xf32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        x_scale = absmax / 127.0
        x_int8 = (xf32 / x_scale).clamp_(-128, 127).to(torch.int8)
        try:
            out_i32 = torch._int_mm(x_int8, weight_kn)
            out_f32 = out_i32.float() * x_scale * self._w_scale.unsqueeze(0)
        except Exception:
            # FlagGems _int_mm may fall back to aten::mm with int32 operands,
            # which re-enters FlagGems mm and fails for non-BF16 dtype.
            # Use an fp32 matmul fallback that bypasses that chain.
            w_fp32 = weight_kn.to(torch.float32) * self._w_scale.unsqueeze(
                0
            )  # [K, N]
            out_f32 = xf32 @ w_fp32  # dynamic quant of x was identity here
        return out_f32.to(torch.bfloat16).reshape(*shape[:-1], self.N)

    def extra_repr(self) -> str:
        return f"in_features={self.K}, out_features={self.N}, dtype=int8"
