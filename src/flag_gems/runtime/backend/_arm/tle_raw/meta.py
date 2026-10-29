# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape-only implementations for the optional Arm inference operators.

Weight preparation and executable-address registration are eager setup
operations. They are deliberately not represented as graph operations.
Mutation and aliasing stay in the C++ schemas; fake kernels never read data.
"""

import threading
import torch

_lock = threading.Lock()
_registered = set()


def _silu_pack(gate_up, mr):
    torch._check(gate_up.ndim == 2)
    torch._check(mr == 1 or mr == 4)
    k = gate_up.shape[1] // 2
    torch._check(gate_up.shape[1] == 2 * k)
    torch._check(k > 0)
    torch._check(k % 32 == 0)
    return gate_up.new_empty(
        ((gate_up.shape[0] + mr - 1) // mr * mr * (k + 8),), dtype=torch.uint8
    )


def register_arm_fake_kernels():
    """Register schemas present in the currently loaded optional library."""
    groups = {"flag_gems_arm_silu_native": {"pack": _silu_pack, "pack_raw": _silu_pack}}
    with _lock:
        for namespace, operations in groups.items():
            for name, implementation in operations.items():
                for overload in ("", "raw", "triton"):
                    qualified = (
                        namespace + "::" + name + ("." + overload if overload else "")
                    )
                    if qualified in _registered:
                        continue
                    try:
                        torch._C._dispatch_find_schema_or_throw(
                            namespace + "::" + name, overload
                        )
                    except RuntimeError:
                        continue
                    if not torch._C._dispatch_has_kernel_for_dispatch_key(
                        qualified, "Meta"
                    ):
                        torch.library.register_fake(qualified)(implementation)
                    _registered.add(qualified)
