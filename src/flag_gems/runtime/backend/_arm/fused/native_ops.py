# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional Arm operators with native and cached CPU TLE execution paths."""

import ctypes
import os
import threading
from functools import lru_cache
from pathlib import Path

import torch

_lock = threading.Lock()
_library = None
_namespaces = {
    "w4a8_g32": "flag_gems_arm_g32",
    "w8a8": "flag_gems_arm_w8a8",
    "moe": "flag_gems_arm_moe_prepack",
    "gdn": "flag_gems_arm_gdn",
}


class _RawOps:
    def __init__(self, namespace):
        self.namespace = namespace

    def __getattr__(self, name):
        operator = getattr(self.namespace, name)
        # Weight preparation, thread policy and partial-output helpers are
        # host functions; complete inference operators expose a Raw overload.
        return operator.raw if hasattr(operator, "raw") else operator


@lru_cache(maxsize=8)
def arm_native_ops(group, *, raw=False):
    """Return a cached operator namespace from an Arm inference build.

    ARM_STANDALONE builds W8A8/GDN without KleidiAI; ARM_KAI also builds
    groupwise W4/MoE. Standalone packed weights must not be passed to KAI.

    ``raw=True`` requires the CPU TLE compiler and initializes compiled entries
    once. Call before inference workers start and cache the needed operator.
    The model adapter supplies shape, thread and SME/Neon split policies.
    """
    global _library
    if group not in _namespaces:
        raise ValueError(f"Unknown Arm native operator group: {group}")
    with _lock:
        if _library is None:
            directory = Path(__file__).resolve().parents[1] / "lib"
            override = os.environ.get("FLAGGEMS_ARM_NATIVE_LIBRARY")
            paths = (
                [Path(override).resolve()]
                if override
                else list(directory.glob("libflag_gems_arm_native.*"))
            )
            if len(paths) != 1 or not paths[0].is_file():
                raise RuntimeError("Build and install ARM_STANDALONE=ON or ARM_KAI=ON first")
            # C ABI symbols must be visible when Triton loads its compiled DSO.
            _library = (paths[0], ctypes.CDLL(str(paths[0]), mode=ctypes.RTLD_GLOBAL))
    if raw:
        from .native_raw import initialize_native_raw

        initialize_native_raw(_library[0])
    namespace = getattr(torch.ops, _namespaces[group])
    return _RawOps(namespace) if raw else namespace
