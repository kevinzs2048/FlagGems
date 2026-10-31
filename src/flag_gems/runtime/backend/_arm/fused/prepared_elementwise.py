# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared BF16 CPU elementwise operators through cached Triton TLE Raw entries."""

import ctypes
import math
import os
import struct
import sys
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call


@triton.jit(do_not_specialize_on_alignment=["descriptor"])
def _elementwise_raw(descriptor, SYMBOL: tl.constexpr):
    raw_call(SYMBOL, descriptor.to(tl.uint64), result_dtype=None)


@lru_cache(maxsize=2)
def _entry(symbol):
    suffix = "dylib" if sys.platform == "darwin" else "so"
    default = (
        Path(__file__).resolve().parents[1]
        / "tle_raw/lib"
        / f"libflag_gems_arm_elementwise.{suffix}"
    )
    path = Path(
        os.environ.get("FLAGGEMS_ARM_ELEMENTWISE_LIBRARY", str(default))
    ).resolve()
    if not path.is_file():
        raise RuntimeError(
            "Build FLAGGEMS_BUILD_ARM_ELEMENTWISE=ON before selecting these operators"
        )
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    getattr(native, symbol)
    compiled = _elementwise_raw.warmup(
        torch.empty(1, dtype=torch.int64), SYMBOL=symbol, grid=(1,), num_cpu_threads=1
    )
    library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    fn = getattr(library, compiled.name)
    fn.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    fn.restype = None
    return native, compiled, library, fn


def _check_input(x):
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Contiguous CPU BF16 input required")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("Inference only")


class PreparedAddLayerNormRaw:
    def __init__(self, weight, bias, eps=1e-5, *, threads=4):
        _check_input(weight.detach())
        _check_input(bias.detach())
        if (
            weight.ndim != 1
            or bias.shape != weight.shape
            or not 1 <= weight.numel() <= 4096
            or not math.isfinite(eps)
            or eps <= 0
            or not isinstance(threads, int)
            or threads < 1
        ):
            raise ValueError(
                "Expected BF16 norm parameters and positive epsilon/threads"
            )
        self.weight, self.bias = weight, bias
        self.n, self.threads = weight.numel(), threads
        self.eps_bits = struct.unpack("I", struct.pack("f", eps))[0]
        self.entry = _entry("flag_gems_arm_add_layer_norm_bf16")

    def __call__(self, x, residual):
        _check_input(x)
        _check_input(residual)
        if x.ndim < 2 or x.shape[-1] != self.n or residual.shape != x.shape:
            raise ValueError(
                "Matching BF16 residual shapes ending in norm width required"
            )
        if torch.is_grad_enabled() and (
            self.weight.requires_grad or self.bias.requires_grad
        ):
            raise RuntimeError("Inference only")
        y = torch.empty_like(x)
        if x.numel():
            descriptor = (ctypes.c_int64 * 9)(
                x.data_ptr(),
                residual.data_ptr(),
                self.weight.data_ptr(),
                self.bias.data_ptr(),
                y.data_ptr(),
                x.numel() // self.n,
                self.n,
                self.threads,
                self.eps_bits,
            )
            self.entry[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
        return y


class PreparedGeluErfRaw:
    def __init__(self, *, threads=8, reference_threads=None):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("Positive thread count required")
        self.threads = threads
        self.reference_threads = reference_threads
        if reference_threads is not None and (
            not isinstance(reference_threads, int) or reference_threads < 1
        ):
            raise ValueError("Positive reference thread count required")
        self.entry = _entry("flag_gems_arm_gelu_lookup_bf16")

    def __call__(self, x):
        _check_input(x)
        y = torch.empty_like(x)
        if x.numel():
            descriptor = (ctypes.c_int64 * 5)(
                x.data_ptr(),
                y.data_ptr(),
                x.numel(),
                self.threads,
                self.reference_threads or torch.get_num_threads(),
            )
            self.entry[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
        return y
