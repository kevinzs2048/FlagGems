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


def _linear(input, packed, n, *options):
    torch._check(input.ndim == 2)
    torch._check(n > 0)
    return input.new_empty((input.shape[0], n), dtype=torch.float32)


def _same_input(input, *args, **kwargs):
    return torch.empty_like(input)


def _value_decode(A_log, a, b, dt_bias, query, key, value, *args):
    return torch.empty_like(value)


def _value_prefill(query, key, value, *args):
    return torch.empty_like(value)


def _mutating(*args, **kwargs):
    # The real operator's schema declares each mutated tensor.
    return None


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


def _moe(input, routing, w1, w2, h, *args):
    return input.new_empty((input.shape[0], h), dtype=torch.float32)


def _moe_decode(input, ids, gates, w1, w2, h, *args):
    return input.new_empty((1, h), dtype=torch.float32)


def register_arm_fake_kernels():
    """Register schemas present in the currently loaded optional library."""
    groups = {
        "flag_gems_arm_w8a8": {
            "linear": _linear,
        },
        "flag_gems_arm_g32": {
            "linear": _linear,
            "moe_decode": _moe_decode,
            **{
                name: _moe
                for name in (
                    "moe_decode_routed",
                    "moe_decode_fused",
                    "moe_prefill_routed",
                    "moe_prefill_fused",
                    "moe_prefill_adaptive",
                    "moe_prefill_balanced",
                )
            },
        },
        "flag_gems_arm_moe_prepack": {
            "moe": _moe,
            "moe_fused": _moe,
        },
        "flag_gems_arm_gdn": {
            "gemma_rmsnorm": _same_input,
            "gemma_fused_add_rmsnorm": _mutating,
            "gdn_decode": _value_decode,
            "gdn_prefill": _value_prefill,
            "gdn_conv1d_update": _same_input,
            "gdn_conv1d_prefill": _same_input,
            "gdn_rmsnorm_gated": _same_input,
            "gdn_packed_decode": _mutating,
            "gdn_packed_decode_norm": _mutating,
        },
        "flag_gems_arm_silu_native": {
            "pack": _silu_pack,
            "pack_raw": _silu_pack,
        },
    }
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
