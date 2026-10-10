# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared ordinary CPU Triton Gemma arithmetic with the ARM rounding order."""
import ctypes
import threading

import torch
import triton
import triton.language as tl

_lock = threading.Lock()
_owners = None
_initialized = False


@triton.jit(do_not_specialize_on_alignment=["x", "w", "r", "y"])
def _gemma_ordered(
    x,
    w,
    r,
    y,
    D: tl.constexpr,
    RESIDUAL: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, 8)
    acc = tl.full((8,), 0, tl.float32)
    for off in range(0, D // 8 * 8, 8):
        value = tl.load(x + row * D + off + lanes).to(tl.float32)
        if RESIDUAL:
            value = (
                (value + tl.load(r + row * D + off + lanes).to(tl.float32))
                .to(r.dtype.element_ty)
                .to(tl.float32)
            )
            tl.store(r + row * D + off + lanes, value)
        acc = acc + value * value
    # Match the two four-lane NEON accumulators and the native reduction tree.
    pair = tl.sum(tl.reshape(acc, (2, 4)), 0)
    ids = tl.arange(0, 4)
    s0 = tl.sum(tl.where(ids == 0, pair, 0), 0)
    s1 = tl.sum(tl.where(ids == 1, pair, 0), 0)
    s2 = tl.sum(tl.where(ids == 2, pair, 0), 0)
    s3 = tl.sum(tl.where(ids == 3, pair, 0), 0)
    total = (s0 + s2) + (s1 + s3)
    for off in range(D // 8 * 8, D):
        value = tl.load(x + row * D + off).to(tl.float32)
        if RESIDUAL:
            value = (
                (value + tl.load(r + row * D + off).to(tl.float32))
                .to(r.dtype.element_ty)
                .to(tl.float32)
            )
            tl.store(r + row * D + off, value)
        total = total + value * value
    scale = 1.0 / tl.sqrt(total / D + EPS)
    cols = tl.arange(0, B)
    for off in range(0, D, B):
        indices = off + cols
        if RESIDUAL:
            value = tl.load(r + row * D + indices, indices < D, 0).to(tl.float32)
        else:
            value = tl.load(x + row * D + indices, indices < D, 0).to(tl.float32)
        weight = tl.load(w + indices, indices < D, 0).to(tl.float32) + 1.0
        tl.store(y + row * D + indices, value * scale * weight, indices < D)


def prepare_gemma_triton():
    """Prepare once before inference; returns the normal Torch operator group.

    The C++ host owns tensors and allocation, while a cached ordinary Triton
    entry computes each row. This path contains no TLE Raw/native math call.
    Supported hidden sizes are 128/1024/2048, BF16, epsilon 1e-6.
    """
    global _owners, _initialized
    from .native_ops import arm_native_ops

    ops = arm_native_ops("gdn")
    with _lock:
        if _owners is not None:
            if not _initialized:
                raise RuntimeError(
                    "Gemma Triton registration failed; restart the process"
                )
            return ops
        prepared = []
        fusion = ops.gemma_triton_fp_fusion()
        for hidden in (128, 1024, 2048):
            x = torch.empty((1, hidden), dtype=torch.bfloat16)
            w = torch.empty(hidden, dtype=torch.bfloat16)
            state, output = torch.empty_like(x), torch.empty_like(x)
            for fused in (False, True):
                compiled = _gemma_ordered.warmup(
                    x,
                    w,
                    state,
                    output,
                    hidden,
                    fused,
                    1e-6,
                    128,
                    grid=(1,),
                    enable_fp_fusion=fusion,
                )
                library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
                address = ctypes.cast(
                    getattr(library, compiled.name), ctypes.c_void_p
                ).value
                prepared.append((hidden, fused, compiled, library, address))
        # Retain all owners before exposing any executable address.
        _owners = prepared
        for hidden, fused, _, _, address in prepared:
            ops.register_gemma_triton_kernel(hidden, fused, address)
        _initialized = True
    return ops
