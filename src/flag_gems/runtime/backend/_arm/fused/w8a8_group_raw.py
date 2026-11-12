# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cached CPU Triton TLE Raw entry for the Arm G32 SDOT compute plan."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call


@triton.jit(do_not_specialize_on_alignment=["descriptor_ptr"])
def _w8g32_sdot_raw_kernel(descriptor_ptr):
    raw_call(
        "flag_gems_arm_w8g32_sdot", descriptor_ptr.to(tl.uint64), result_dtype=None
    )


@lru_cache(maxsize=1)
def _entry():
    suffix = "dylib" if sys.platform == "darwin" else "so"
    default = (
        Path(__file__).resolve().parents[1] / "lib" / f"libflag_gems_arm_w8g32.{suffix}"
    )
    path = Path(os.getenv("FLAGGEMS_ARM_W8G32_LIBRARY", str(default))).resolve()
    if not path.is_file():
        raise RuntimeError(
            "Build FlagGems ARM_STANDALONE (flag_gems_arm_w8g32 target) before selecting raw W8 G32"
        )
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    dummy = torch.empty(1, dtype=torch.uint8)
    compiled = _w8g32_sdot_raw_kernel.warmup(dummy, grid=(1,), num_cpu_threads=1)
    dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    function = getattr(dso, compiled.name)
    # Existing synchronous CPU Triton ABI: descriptor, three IDs, three sizes.
    function.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    function.restype = None
    return native, compiled, dso, function


def run_sdot(
    q, activation_scales, weight, weight_scales, output, threads, grouped_input
):
    m, k = q.shape
    n = weight.shape[0]
    descriptor = torch.tensor(
        [
            q.data_ptr(),
            activation_scales.data_ptr(),
            weight.data_ptr(),
            weight_scales.data_ptr(),
            output.data_ptr(),
            m,
            n,
            k,
            min(threads, ((m + 3) // 4) * ((n + 3) // 4)),
            int(grouped_input),
        ],
        dtype=torch.int64,
        device="cpu",
    )
    # The cached entry is still the compiled Triton kernel, not the C compute
    # function. Its TLE Raw call surrounds the complete parallel compute plan.
    # Keep tensors and DSOs alive until this synchronous entry returns.
    _entry()[3](descriptor.data_ptr(), 0, 0, 0, 1, 1, 1)
