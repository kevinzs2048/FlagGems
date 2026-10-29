# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused SiLU/A8 through a cached CPU TLE entry and native Torch operator."""

import ctypes
import os
import threading

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call

from .silu_mul_a8_pack_native import _ensure_native_library

_lock = threading.Lock()
_owners = None


@triton.jit(
    do_not_specialize=["M", "STRIDE", "K", "MR"],
    do_not_specialize_on_alignment=["x", "out", "status"],
)
def _silu_a8_raw_kernel(
    x,
    out,
    M,
    STRIDE,
    K,
    MR,
    status,
    MR_CODE: tl.constexpr = 0,
    ROW_ONE: tl.constexpr = 0,
):
    if ROW_ONE:
        selected_m = tl.full((), 1, tl.int32)
    else:
        selected_m = M.to(tl.int32)
    if MR_CODE:
        selected_mr = tl.full((), MR_CODE, tl.int32)
    else:
        selected_mr = MR.to(tl.int32)
    result = raw_call(
        "flag_gems_arm_silu_mul_a8_pack_safe",
        x.to(tl.uint64),
        out.to(tl.uint64),
        selected_m,
        STRIDE.to(tl.int32),
        K.to(tl.int32),
        selected_mr,
        result_dtype=tl.int32,
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

        version = getattr(
            _global_native_library, "flag_gems_arm_silu_raw_abi_version", None
        )
        if version is None:
            raise RuntimeError(
                "Rebuild the native SiLU library: missing Raw ABI version"
            )
        version.argtypes = []
        version.restype = ctypes.c_uint32
        if version() != 1:
            raise RuntimeError("Rebuild the native SiLU library: incompatible Raw ABI")
        x = torch.empty(1, 64, device="cpu")
        out = torch.empty(72, dtype=torch.uint8, device="cpu")
        status = torch.empty(1, dtype=torch.int32, device="cpu")
        bitcode = os.environ.get("FLAGGEMS_ARM_SILU_BITCODE")
        specialize_mr = (
            bool(bitcode)
            and os.environ.get("FLAGGEMS_ARM_SILU_SPECIALIZE_MR", "1") == "1"
        )
        variants = []
        configurations = ((1, 1), (1, 4), (0, 0)) if specialize_mr else ((0, 0),)
        for row_one, mr_code in configurations:
            linked = bitcode if row_one or not specialize_mr else None
            compiled = _silu_a8_raw_kernel.warmup(
                x,
                out,
                1,
                64,
                32,
                mr_code or 1,
                status,
                MR_CODE=mr_code,
                ROW_ONE=row_one,
                grid=(1,),
                enable_single_program_entry=True,
                single_program_return_type="i32",
                extern_libs={"arm_silu": linked} if linked else None,
            )
            entry_name = getattr(compiled.metadata, "single_program_name", None)
            if (
                entry_name is None
                or getattr(compiled.metadata, "single_program_return_type", None)
                != "i32"
            ):
                raise RuntimeError(
                    "Rebuild FlagTree CPU with single-program scalar return support"
                )
            dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
            pointer = ctypes.cast(getattr(dso, entry_name), ctypes.c_void_p).value
            variants.append((mr_code, compiled, dso, pointer))
        if specialize_mr:
            register = _global_native_library.flag_gems_arm_register_silu_raw_variant
            register.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
            register.restype = None
            for mr_code, _, _, pointer in variants:
                register(mr_code, pointer)
        else:
            register = _global_native_library.flag_gems_arm_register_silu_raw_kernel
            register.argtypes = [ctypes.c_uint64]
            register.restype = None
            register(variants[0][3])
        # Publish readiness only after every C++ address is visible. ctypes
        # may release the GIL during registration; other callers must wait.
        _owners = (_global_native_library, variants[0][1], variants[0][2], variants)


def silu_mul_a8_pack_raw(gate_up: torch.Tensor, mr: int = 4) -> torch.Tensor:
    """CPU FP32 [M,2*K] -> padded A8 panels, via cached C++ launch of Triton."""
    initialize_silu_a8_raw()
    return torch.ops.flag_gems_arm_silu_native.pack_raw.default(gate_up, mr)
