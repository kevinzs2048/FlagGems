# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental two W8 GEMMs -> exact BF16 SwiGLU -> packed BF16 LHS.

Shares the validated W8 packed ABI. No full-sized projection intermediates.
Negative performance experiment: do not enable in the model. The row-store
version is retained for compiler diagnostics; requires local full-K lowering.
"""
import torch
import triton
import triton.language as tl
from flag_gems.utils import triton_lang_extension as tle

from .w8a8_packed_sme2 import _accumulator_row, _validate


@triton.jit
def _bf16_bits(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where(value != value, 0x7FC0, rounded).to(tl.uint16)


@triton.jit
def _fused_kernel(
    A,
    AI,
    AF,
    WG,
    WU,
    SG,
    SU,
    DG,
    DU,
    Table,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    MR: tl.constexpr,
    CHUNK: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
):
    rows16 = tl.cdiv(M, MR) * (MR // 16)
    begin = tle.program_id(0) * CHUNK
    end = tl.minimum(begin + CHUNK, rows16 * (N // 16))
    lanes = tl.arange(0, 2048)
    cols16 = tl.arange(0, 16)
    for job in tl.range(begin, end, loop_unroll_factor=1):
        if COLUMN_MAJOR:
            mi, ni = job % rows16, job // rows16
        else:
            mi, ni = job // (N // 16), job % (N // 16)
        cg = tl.zeros((16, 16), tl.int32)
        cu = tl.zeros((16, 16), tl.int32)
        for group in tl.range(0, K // 128, loop_unroll_factor=1):
            # Reuse the last valid A8 panel for output-only padding rows;
            # the epilogue writes zeros there. Keep this load unmasked so
            # the validated packed full-K SME2 matcher can recognize it.
            ami = tl.minimum(mi, tl.cdiv(M, 16) - 1)
            a = tl.load(A + ami * 16 * (K + 8) + group * 2048 + lanes).reshape(
                32, 16, 4
            )
            g = tl.load(WG + ni * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
            u = tl.load(WU + ni * 16 * K + group * 2048 + lanes).reshape(32, 16, 4)
            a = a.permute(1, 0, 2).reshape(16, 128)
            g = g.permute(0, 2, 1).reshape(128, 16)
            u = u.permute(0, 2, 1).reshape(128, 16)
            cg += tl.dot(a, g, out_dtype=tl.int32)
            cu += tl.dot(a, u, out_dtype=tl.int32)
        cols = ni * 16 + cols16
        sumg, sumu = tl.load(SG + cols), tl.load(SU + cols)
        dg, du = tl.load(DG + cols), tl.load(DU + cols)
        for row in tl.static_range(16):
            r = mi * 16 + row
            meta = mi * 4 * (K + 8) + 4 * K + row
            zp = tl.load(AI + meta, r < M, other=0)
            scale = tl.load(AF + meta + 16, r < M, other=0)
            g = (_accumulator_row(cg, row) + zp * sumg).to(tl.float32) * (scale * dg)
            u = (_accumulator_row(cu, row) + zp * sumu).to(tl.float32) * (scale * du)
            gb, ub = _bf16_bits(g), _bf16_bits(u)
            sb = tl.load(Table + gb.to(tl.int32)).to(tl.uint16)
            sf = (sb.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
            uf = (ub.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
            out = tl.where(r < M, _bf16_bits(sf * uf), 0).to(tl.uint16)
            offset = (r // MR) * MR * N + (cols // 2) * 2 * MR + (r % MR) * 2 + cols % 2
            tl.store(Out + offset, out)


def fused_swiglu_packed(
    lhs, gate, up, table, m, *, packed_rows, workers=12, chunk=16, column_major=False
):
    """gate/up are PackedWeight-compatible objects; return native BF16 ABI bytes."""
    if (gate.n, gate.k) != (up.n, up.k):
        raise ValueError("Projection dimensions must match")
    if packed_rows not in (32, 64) or chunk not in (1, 2, 4, 8, 16):
        raise ValueError("Unvalidated BF16 output ABI or task chunk")
    for weight in (gate, up):
        _validate(
            lhs,
            weight.data,
            weight.sums,
            weight.scales,
            m,
            weight.n,
            weight.k,
            workers,
            chunk,
            16,
        )
    tensors = (
        lhs,
        gate.data,
        gate.sums,
        gate.scales,
        up.data,
        up.sums,
        up.scales,
        table,
    )
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("Expected CPU tensors")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise ValueError("Inference only")
    if (
        table.dtype != torch.bfloat16
        or table.shape != (65536,)
        or not table.is_contiguous()
    ):
        raise ValueError("Expected complete BF16 SiLU table")
    n, k = gate.n, gate.k
    padded_m = triton.cdiv(m, packed_rows) * packed_rows
    out = torch.empty(padded_m * n, dtype=torch.int16)
    if m:
        jobs = padded_m // 16 * (n // 16)
        _fused_kernel[(triton.cdiv(jobs, chunk),)](
            lhs.view(torch.int8),
            lhs.view(torch.int32),
            lhs.view(torch.float32),
            gate.data,
            up.data,
            gate.sums,
            up.sums,
            gate.scales,
            up.scales,
            table.view(torch.int16),
            out,
            m,
            n,
            k,
            packed_rows,
            chunk,
            column_major,
            num_cpu_threads=workers,
            enable_sme2=True,
            sme_vector_bits=512,
            enable_fast_math=False,
            enable_fp_fusion=False,
        )
    return out.view(torch.uint8)
