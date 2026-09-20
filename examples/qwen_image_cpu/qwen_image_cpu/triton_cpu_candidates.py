"""Experimental generated CPU kernels; never enabled by the inference preset.

Keep this separate from native SME2 baselines until model accuracy and paired
end-to-end timing pass. The integer GEMM is a capability probe, not a W8 Linear.
"""

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle


@triton.jit
def _round_bf16_bits(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0x7FC0, rounded).to(tl.uint16)


@triton.jit
def _residual_kernel(
    X,
    Y,
    G,
    Output,
    S: tl.constexpr,
    D: tl.constexpr,
    GS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tle.program_id(0)
    gate_row = row // S if GS == 1 else row
    for start in range(0, D, TILE):
        col = start + tl.arange(0, TILE)
        x = tl.load(X + row * D + col, col < D, 0).to(tl.uint32)
        y = tl.load(Y + row * D + col, col < D, 0).to(tl.uint32)
        g = tl.load(G + gate_row * D + col, col < D, 0).to(tl.uint32)
        xf = (x << 16).to(tl.float32, bitcast=True)
        yf = (y << 16).to(tl.float32, bitcast=True)
        gf = (g << 16).to(tl.float32, bitcast=True)
        product = (_round_bf16_bits(yf * gf).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        tl.store(Output + row * D + col, _round_bf16_bits(xf + product), col < D)


def residual(x, branch, gate, *, workers=18, tile=128):
    if (
        x.device.type != "cpu"
        or branch.device.type != "cpu"
        or gate.device.type != "cpu"
        or x.dtype != torch.bfloat16
        or branch.dtype != x.dtype
        or gate.dtype != x.dtype
        or x.ndim != 3
        or x.shape != branch.shape
        or x.shape[-1] <= 0
        or gate.ndim != 3
        or gate.shape[0] != x.shape[0]
        or gate.shape[-1] != x.shape[-1]
        or gate.shape[1] not in (1, x.shape[1])
    ):
        raise ValueError(
            "Expected matching CPU BF16 [B,S,D] inputs and broadcastable gate"
        )
    if type(workers) is not int or workers < 1 or tile not in (64, 128, 256, 512):
        raise ValueError("Invalid candidate launch configuration")
    x, branch, gate = x.contiguous(), branch.contiguous(), gate.contiguous()
    out = torch.empty_like(x)
    if x.numel():
        _residual_kernel[(x.shape[0] * x.shape[1],)](
            x.view(torch.uint16),
            branch.view(torch.uint16),
            gate.view(torch.uint16),
            out.view(torch.uint16),
            x.shape[1],
            x.shape[2],
            gate.shape[1],
            tile,
            num_cpu_threads=workers,
            enable_fast_math=False,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _w8_integer_kernel(A, W, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tle.program_id(0) * 16 + tl.arange(0, 16)
    cols = tle.program_id(1) * 16 + tl.arange(0, 16)
    ki = tl.arange(0, 128)
    total = tl.zeros((16, 16), tl.int32)
    for start in tl.range(0, K, 128, loop_unroll_factor=1):
        kk = start + ki
        a = tl.load(
            A + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            0,
        )
        w = tl.load(
            W + cols[None, :] * K + kk[:, None],
            (cols[None, :] < N) & (kk[:, None] < K),
            0,
        )
        # The current SME2 lowering recognizes a zero-initialized 16x128 dot.
        # Keep the full-K sum in integer arithmetic, not groupwise FP32.
        total += tl.dot(a, w, out_dtype=tl.int32)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        total,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def integer_gemm(a, w, *, workers=18):
    if (
        a.device.type != "cpu"
        or w.device.type != "cpu"
        or a.dtype != torch.int8
        or w.dtype != torch.int8
        or a.ndim != 2
        or w.ndim != 2
        or a.shape[1] != w.shape[1]
        or not a.is_contiguous()
        or not w.is_contiguous()
        or not 0 < a.shape[1] <= 32768
        or type(workers) is not int
        or workers < 1
    ):
        raise ValueError("Expected contiguous CPU INT8 [M,K], [N,K], 0<K<=32768")
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.int32)
    if m and n:
        _w8_integer_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 16))](
            a,
            w,
            out,
            m,
            n,
            k,
            num_cpu_threads=workers,
            enable_sme2=True,
            enable_fast_math=False,
            enable_fp_fusion=False,
        )
    return out


def pack_integer_panels(codes):
    """Portable lossless [rows,K] -> [rows/16,K/128,K4,rows16,4]."""
    if (
        codes.device.type != "cpu"
        or codes.dtype != torch.int8
        or codes.ndim != 2
        or codes.shape[0] % 16
        or codes.shape[1] % 128
    ):
        raise ValueError("Packed probe requires CPU INT8 rows%16=0, K%128=0")
    rows, k = codes.shape
    return (
        codes.reshape(rows // 16, 16, k // 128, 32, 4)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )


@triton.jit
def _w8_integer_packed_kernel(
    A, W, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    mi, ni = tle.program_id(0), tle.program_id(1)
    lanes = tl.arange(0, 2048)
    rows, cols = tl.arange(0, 16), tl.arange(0, 16)
    total = tl.zeros((16, 16), tl.int32)
    for group in tl.range(0, K // 128, loop_unroll_factor=1):
        av = tl.load(A + mi * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
        wv = tl.load(W + ni * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
        a = av.permute(1, 0, 2).reshape(16, 128)
        w = wv.permute(0, 2, 1).reshape(128, 16)
        total += tl.dot(a, w, out_dtype=tl.int32)
    tl.store(C + (mi * 16 + rows[:, None]) * N + ni * 16 + cols[None, :], total)


@triton.jit
def _w8_integer_packed_four_kernel(
    A, W, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    mi, ni = tle.program_id(0) * 2, tle.program_id(1) * 2
    lanes = tl.arange(0, 2048)
    rows, cols = tl.arange(0, 16), tl.arange(0, 16)
    c00 = tl.zeros((16, 16), tl.int32)
    c01 = tl.zeros((16, 16), tl.int32)
    c10 = tl.zeros((16, 16), tl.int32)
    c11 = tl.zeros((16, 16), tl.int32)
    for group in tl.range(0, K // 128, loop_unroll_factor=1):
        a0 = tl.load(A + mi * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
        a1 = tl.load(A + (mi + 1) * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
        b0 = tl.load(W + ni * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
        b1 = tl.load(W + (ni + 1) * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
        a0 = a0.permute(1, 0, 2).reshape(16, 128)
        a1 = a1.permute(1, 0, 2).reshape(16, 128)
        b0 = b0.permute(0, 2, 1).reshape(128, 16)
        b1 = b1.permute(0, 2, 1).reshape(128, 16)
        c00 += tl.dot(a0, b0, out_dtype=tl.int32)
        c01 += tl.dot(a0, b1, out_dtype=tl.int32)
        c10 += tl.dot(a1, b0, out_dtype=tl.int32)
        c11 += tl.dot(a1, b1, out_dtype=tl.int32)
    offset = (mi * 16 + rows[:, None]) * N + ni * 16 + cols[None, :]
    tl.store(C + offset, c00)
    tl.store(C + offset + 16, c01)
    tl.store(C + offset + 16 * N, c10)
    tl.store(C + offset + 16 * N + 16, c11)


def integer_gemm_packed(a, w, *, workers=18, tile=16):
    if (
        a.device.type != "cpu"
        or w.device.type != "cpu"
        or a.dtype != torch.int8
        or w.dtype != torch.int8
        or a.ndim != 5
        or w.ndim != 5
        or a.shape[2:] != (32, 16, 4)
        or w.shape[2:] != (32, 16, 4)
        or a.shape[1] != w.shape[1]
        or not 0 < a.shape[1] <= 256
        or not a.is_contiguous()
        or not w.is_contiguous()
        or type(workers) is not int
        or workers < 1
    ):
        raise ValueError("Expected packed CPU INT8 panels with 0<K<=32768")
    m, n, k = a.shape[0] * 16, w.shape[0] * 16, a.shape[1] * 128
    if tile not in (16, 32) or m % tile or n % tile:
        raise ValueError("Packed tile must divide both M and N")
    out = torch.empty((m, n), dtype=torch.int32)
    if m and n:
        kernel = (
            _w8_integer_packed_kernel if tile == 16 else _w8_integer_packed_four_kernel
        )
        kernel[(m // tile, n // tile)](
            a,
            w,
            out,
            m,
            n,
            k,
            num_cpu_threads=workers,
            enable_sme2=True,
            enable_fast_math=False,
            enable_fp_fusion=False,
        )
    return out
