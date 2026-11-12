# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 residual addition followed by LayerNorm, preserving the add rounding."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _add_layer_norm(X, R, W, B, Y, EPS: tl.constexpr, N: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row*N + col, mask, 0).to(tl.float32)
    r = tl.load(R + row*N + col, mask, 0).to(tl.float32)
    # ATen materializes the residual sum in BF16 before LayerNorm.
    v = (x+r).to(tl.bfloat16).to(tl.float32)
    mean = tl.sum(tl.where(mask, v, 0), 0)/N
    centered = tl.where(mask, v-mean, 0)
    variance = tl.sum(centered*centered, 0)/N
    w = tl.load(W+col, mask, 0).to(tl.float32)
    b = tl.load(B+col, mask, 0).to(tl.float32)
    tl.store(Y+row*N+col, centered*tl.rsqrt(variance+EPS)*w+b, mask)


def add_layer_norm(x, residual, weight, bias, eps=1e-5, *, threads=8):
    if (x.device.type != 'cpu' or x.dtype != torch.bfloat16 or x.ndim < 2
            or not x.is_contiguous() or residual.shape != x.shape
            or residual.device != x.device or residual.dtype != x.dtype
            or not residual.is_contiguous() or not 1 <= x.shape[-1] <= 4096):
        raise ValueError('Expected matching contiguous CPU BF16 inputs, width <=4096')
    n=x.shape[-1]
    if any(t.device != x.device or t.dtype != x.dtype or t.shape != (n,)
           or not t.is_contiguous() for t in (weight,bias)):
        raise ValueError('Expected contiguous BF16 weight and bias [N]')
    if not math.isfinite(eps) or eps <= 0 or not isinstance(threads,int) or threads < 1:
        raise ValueError('Positive finite epsilon and thread count required')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x,residual,weight,bias)):
        raise RuntimeError('Inference only')
    y=torch.empty_like(x)
    rows=x.numel()//n
    if rows:
        _add_layer_norm[(rows,)](x,residual,weight,bias,y,EPS=eps,N=n,
            BLOCK=triton.next_power_of_2(n),num_cpu_threads=threads,enable_fp_fusion=False)
    return y
