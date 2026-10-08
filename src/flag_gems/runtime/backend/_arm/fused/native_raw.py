# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Synchronous CPU TLE entry for native Arm tensor operations.

Initialize once, then use the native Torch operator's ``.raw`` overload.
Tensor ownership, validation, exceptions and parallel work remain in C++.
The compiled DSO is retained for the lifetime of the registered address.
"""

import ctypes
import threading

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call

_lock = threading.Lock()
_owners = None


@triton.jit(do_not_specialize_on_alignment=["descriptor_ptr"])
def _arm_native_raw_kernel(descriptor_ptr, SYMBOL: tl.constexpr):
    raw_call(
        SYMBOL,
        descriptor_ptr.to(tl.uint64),
        result_dtype=None,
    )


def initialize_native_raw(library_path):
    """Register a compiled CPU entry; call before starting parallel inference."""
    global _owners
    with _lock:
        if _owners is not None:
            if _owners[0] != str(library_path):
                raise RuntimeError(
                    "CPU Raw entry already registered with another library"
                )
            return
        native = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        dummy = torch.empty(1, dtype=torch.uint8, device="cpu")
        compiled = _arm_native_raw_kernel.warmup(
            dummy, SYMBOL="flag_gems_arm_raw_dispatch", grid=(1,)
        )
        kernel = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
        address = ctypes.cast(getattr(kernel, compiled.name), ctypes.c_void_p).value
        register = native.flag_gems_arm_register_raw_kernel
        register.argtypes = [ctypes.c_uint64]
        register.restype = None
        register_dense = native.flag_gems_arm_register_dense_raw_kernel
        register_dense.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
        register_dense.restype = None
        dense_owners = []
        for index, symbol in enumerate(
            ("flag_gems_arm_g32_compute", "flag_gems_arm_w8_compute")
        ):
            entry = _arm_native_raw_kernel.warmup(dummy, SYMBOL=symbol, grid=(1,))
            dso = ctypes.CDLL(entry.metadata_group[entry.name + ".so"])
            pointer = ctypes.cast(getattr(dso, entry.name), ctypes.c_void_p).value
            dense_owners.append((entry, dso, pointer))
        _owners = (str(library_path), native, compiled, kernel, dense_owners)
        register(address)
        for index, (_, _, pointer) in enumerate(dense_owners):
            register_dense(index, pointer)
