# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional C++/NEON fused FP32 SiLU*up and asymmetric A8 KAI packing.

This is a direct native operator, not a TLE Raw implementation. Build with
FLAGGEMS_BUILD_ARM_NATIVE=ON and install the library beside the Arm backend.
"""

import ctypes
import os
import threading
from pathlib import Path

import torch

_load_lock = threading.Lock()
_global_native_library = None


def _ensure_native_library(global_symbols=False):
    global _global_native_library
    # Initialization is synchronized; steady-state calls need no loader lock.
    if global_symbols and _global_native_library is not None:
        return
    if not global_symbols and hasattr(torch.ops.flag_gems_arm_silu_native, "pack"):
        return
    with _load_lock:
        if global_symbols and _global_native_library is not None:
            return
        if not global_symbols and hasattr(torch.ops.flag_gems_arm_silu_native, "pack"):
            return
        if global_symbols:
            # Promote an already loaded operator library instead of registering
            # the same Torch namespace again from a different build artifact.
            process = ctypes.CDLL(None)
            if hasattr(process, "flag_gems_arm_register_silu_raw_kernel"):
                _global_native_library = process
                return
            for loaded_path in torch.ops.loaded_libraries:
                library = ctypes.CDLL(loaded_path)
                if hasattr(library, "flag_gems_arm_register_silu_raw_kernel"):
                    _global_native_library = ctypes.CDLL(
                        loaded_path, mode=ctypes.RTLD_GLOBAL
                    )
                    return
        directory = Path(__file__).resolve().parents[1] / "lib"
        override = os.environ.get("FLAGGEMS_ARM_SILU_LIBRARY")
        libraries = (
            [Path(override).resolve()]
            if override
            else list(directory.glob("libflag_gems_arm_silu_native.*"))
        )
        if len(libraries) != 1 or not libraries[0].is_file():
            raise RuntimeError("Build and install FLAGGEMS_BUILD_ARM_NATIVE=ON first")
        if global_symbols:
            _global_native_library = ctypes.CDLL(
                str(libraries[0]), mode=ctypes.RTLD_GLOBAL
            )
        else:
            torch.ops.load_library(str(libraries[0]))


def silu_mul_a8_pack_native(gate_up: torch.Tensor, mr: int = 4) -> torch.Tensor:
    """Pack CPU FP32 [M,2*K], K multiple of 32, MR 1 or 4; RNE codes.

    Caller supplies finite values with a representable activation range and
    inner stride one. Output holds padded KAI panels and row scale/offset.
    """
    _ensure_native_library()
    return torch.ops.flag_gems_arm_silu_native.pack(gate_up, mr)
