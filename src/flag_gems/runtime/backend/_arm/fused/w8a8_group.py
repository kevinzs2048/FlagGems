# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU Triton W8A8 reference with symmetric G32 weight scales.

Activation scales are dynamic per token or per token/group, as declared by
the checkpoint. Integer products accumulate within individual weight groups
before applying their FP32 scales.
"""

import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quantize_token(
    X, Q, S, Z, K: tl.constexpr, BK: tl.constexpr, SYMMETRIC: tl.constexpr
):
    row = tl.program_id(0)
    j = tl.arange(0, BK)
    x = tl.load(X + row * K + j, j < K, 0).to(tl.float32)
    if SYMMETRIC:
        peak = tl.max(tl.abs(x), 0)
        scale = tl.where(peak > 0, peak / 127.0, 1.0)
        q = tl.minimum(tl.maximum(libdevice.rint(x / scale), -127.0), 127.0).to(tl.int8)
    else:
        low = tl.minimum(tl.min(x, 0), 0.0)
        high = tl.maximum(tl.max(x, 0), 0.0)
        scale = tl.where(high > low, (high - low) / 255.0, 1.0)
        zero = tl.minimum(
            tl.maximum(libdevice.rint(-128.0 - low / scale), -128.0), 127.0
        )
        q = tl.minimum(tl.maximum(libdevice.rint(x / scale) + zero, -128.0), 127.0).to(
            tl.int8
        )
        tl.store(Z + row, zero.to(tl.int32))
    tl.store(Q + row * K + j, q, j < K)
    tl.store(S + row, scale)


@triton.jit
def _quantize_groups(X, Q, S, K: tl.constexpr, BG: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1) * BG + tl.arange(0, BG)
    offset = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * K + offset, offset < K, 0).to(tl.float32)
    peak = tl.max(tl.abs(x), 1)
    scale = tl.where(peak > 0, peak / 127.0, 1.0)
    q = tl.minimum(tl.maximum(libdevice.rint(x / scale[:, None]), -127.0), 127.0).to(
        tl.int8
    )
    tl.store(Q + row * K + offset, q, offset < K)
    tl.store(S + row * (K // 32) + group, scale, group < K // 32)


@triton.jit
def _linear_g32(
    Q,
    XS,
    W,
    WS,
    Y,
    Z,
    SYMMETRIC: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    INPUT_GROUPED: tl.constexpr = False,
):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    row = tl.program_id(1)
    result = tl.full((BN,), 0.0, tl.float32)
    local = tl.arange(0, BK)
    local_group = tl.arange(0, BK // 32)
    for tile in range(triton.cdiv(K, BK)):
        j = tile * BK + local
        w = tl.load(
            W + cols[:, None] * K + j[None, :],
            (cols[:, None] < N) & (j[None, :] < K),
            0,
        ).to(tl.int32)
        x = tl.load(Q + row * K + j, j < K, 0).to(tl.int32)
        if not SYMMETRIC:
            x -= tl.load(Z + row)
        products = w * x[None, :]
        grouped = tl.sum(tl.reshape(products, (BN, BK // 32, 32)), 2).to(tl.float32)
        group = tile * (BK // 32) + local_group
        scale = tl.load(
            WS + cols[:, None] * (K // 32) + group[None, :],
            (cols[:, None] < N) & (group[None, :] < K // 32),
            0,
        )
        if INPUT_GROUPED:
            activation_scale = tl.load(XS + row * (K // 32) + group, group < K // 32, 0)
            scale *= activation_scale[None, :]
        result += tl.sum(grouped * scale, 1)
    if not INPUT_GROUPED:
        result *= tl.load(XS + row)
    tl.store(Y + row * N + cols, result, cols < N)


def linear_w8a8_g32(
    x,
    weight,
    scales,
    bias=None,
    *,
    input_symmetric=True,
    input_group_size=None,
    prepared_weight=None,
):
    if (
        x.device.type != "cpu"
        or weight.device.type != "cpu"
        or scales.device.type != "cpu"
    ):
        raise ValueError("CPU tensors required")
    if weight.dtype != torch.int8 or weight.ndim != 2 or not weight.is_contiguous():
        raise ValueError("Expected contiguous int8 weight [N,K]")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Floating point activations required")
    n, k = weight.shape
    if k % 32 or x.shape[-1] != k or tuple(scales.shape) != (n, k // 32):
        raise ValueError("G32 shape mismatch")
    if scales.dtype != torch.float32 or not scales.is_contiguous():
        raise ValueError("Expected contiguous FP32 group scales")
    shape = x.shape[:-1]
    dtype = x.dtype
    x = x.reshape(-1, k).contiguous()
    m = x.shape[0]
    if m == 0 or n == 0:
        return x.new_empty((*shape, n))
    threads = int(os.environ.get("OMP_NUM_THREADS", str(torch.get_num_threads())))
    if threads < 1:
        raise ValueError("Positive thread count required")
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    if input_group_size not in (None, 32):
        raise ValueError("Only token A8 or group-32 A8 supported")
    grouped_input = input_group_size == 32
    if grouped_input and not input_symmetric:
        raise ValueError("Group A8 requires symmetric activations")
    xs = torch.empty(
        m * (k // 32 if grouped_input else 1), dtype=torch.float32, device=x.device
    )
    y = torch.empty((m, n), dtype=torch.float32, device=x.device)
    zero = xs if input_symmetric else torch.empty(m, dtype=torch.int32, device=x.device)
    if grouped_input:
        _quantize_groups[(m, triton.cdiv(k // 32, 8))](
            x,
            q,
            xs,
            k,
            8,
            enable_fp_fusion=False,
            num_cpu_threads=min(threads, max(1, m * triton.cdiv(k // 32, 8))),
        )
    else:
        _quantize_token[(m,)](
            x,
            q,
            xs,
            zero,
            k,
            triton.next_power_of_2(k),
            input_symmetric,
            enable_fp_fusion=False,
            num_cpu_threads=min(threads, max(1, m)),
        )
    backend = os.getenv("FLAGGEMS_ARM_W8G32_BACKEND", "triton")
    if backend not in {"triton", "raw", "i8mm"}:
        raise ValueError("FLAGGEMS_ARM_W8G32_BACKEND must be triton, raw or i8mm")
    if backend == "i8mm" and input_symmetric and m >= 64 and n >= 128 and k >= 32:
        from .w8a8_group_i8mm import prepare_i8mm, run_i8mm

        prepared = (
            prepared_weight
            if prepared_weight is not None
            else prepare_i8mm(weight, scales, threads)
        )
        prepared.validate(weight, scales)
        run_i8mm(q, xs, prepared, y, threads, grouped_input)
    elif backend == "raw" and input_symmetric and m >= 64 and n >= 128 and k >= 32:
        from .w8a8_group_raw import run_sdot

        run_sdot(q, xs, weight, scales, y, threads, grouped_input)
    else:
        _linear_g32[(triton.cdiv(n, 4), m)](
            q,
            xs,
            weight,
            scales,
            y,
            zero,
            input_symmetric,
            n,
            k,
            256,
            4,
            grouped_input,
            enable_fp_fusion=False,
            num_cpu_threads=min(threads, max(1, m * triton.cdiv(n, 4))),
        )
    y = y.reshape(*shape, n)
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype)
