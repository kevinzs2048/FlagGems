# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused SiLU/A8 through a cached CPU TLE entry and native Torch operator."""

import ctypes
import threading

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call

from .silu_mul_a8_pack_native import _ensure_native_library

_lock = threading.Lock()
_owners = None


@triton.jit(do_not_specialize=["M", "STRIDE", "K", "MR"])
def _silu_a8_raw_kernel(x, out, M, STRIDE, K, MR, status):
    result = raw_call(
        "flag_gems_arm_silu_mul_a8_pack_addr",
        x.to(tl.uint64),
        out.to(tl.uint64),
        M.to(tl.int32),
        STRIDE.to(tl.int32),
        K.to(tl.int32),
        MR.to(tl.int32),
    )
    tl.store(status, result)


def initialize_silu_a8_raw():
    """Compile once; keep the native library and compiled CPU module alive."""
    global _owners
    if _owners is not None:
        return
    with _lock:
        if _owners is not None:
            return
        _ensure_native_library(global_symbols=True)
        from .silu_mul_a8_pack_native import _global_native_library

        x = torch.empty(1, 64, device="cpu")
        out = torch.empty(72, dtype=torch.uint8, device="cpu")
        status = torch.empty(1, dtype=torch.int32, device="cpu")
        compiled = _silu_a8_raw_kernel.warmup(x, out, 1, 64, 32, 1, status, grid=(1,))
        dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
        pointer = ctypes.cast(getattr(dso, compiled.name), ctypes.c_void_p).value
        register = _global_native_library.flag_gems_arm_register_silu_raw_kernel
        register.argtypes = [ctypes.c_uint64]
        register.restype = None
        _owners = (_global_native_library, compiled, dso)
        register(pointer)


def silu_mul_a8_pack_raw(gate_up: torch.Tensor, mr: int = 4) -> torch.Tensor:
    """CPU FP32 [M,2*K] -> padded A8 panels, via cached C++ launch of Triton."""
    initialize_silu_a8_raw()
    return torch.ops.flag_gems_arm_silu_native.pack.raw(gate_up, mr)
