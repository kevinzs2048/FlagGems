# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import os
import platform

import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("arm64", "aarch64")
    or not os.getenv("FLAGGEMS_ARM_NATIVE_LIBRARY"),
    reason="requires the optional ARM native host build and CPU Triton",
)


@pytest.mark.parametrize("hidden", [128, 1024, 2048])
@pytest.mark.parametrize("rows", [0, 1, 16, 512])
@pytest.mark.parametrize("fused", [False, True])
def test_prepared_gemma_matches_native(hidden, rows, fused):
    from flag_gems.runtime.backend._arm.fused.gemma_norm import prepare_gemma_triton

    ops = prepare_gemma_triton()
    assert prepare_gemma_triton() is ops
    # Offset views must not inherit alignment assumptions from warmup tensors.
    x = torch.randn(rows * hidden + 1, dtype=torch.bfloat16)[1:].view(rows, hidden)
    weight = torch.randn(hidden + 1, dtype=torch.bfloat16)[1:]
    residual = torch.randn_like(x)
    if fused:
        expected, expected_state = x.clone(), residual.clone()
        ops.gemma_fused_add_rmsnorm(expected, expected_state, weight, 1e-6)
        actual = torch.empty(x.numel() + 1, dtype=x.dtype)[1:].view_as(x)
        state = torch.empty(x.numel() + 1, dtype=x.dtype)[1:].view_as(x)
        actual.copy_(x)
        state.copy_(residual)
        ops.gemma_fused_add_rmsnorm.triton(actual, state, weight, 1e-6)
        assert torch.equal(actual, expected) and torch.equal(state, expected_state)
    else:
        expected = ops.gemma_rmsnorm(x, weight, 1e-6)
        assert torch.equal(ops.gemma_rmsnorm.triton(x, weight, 1e-6), expected)


def test_prepared_gemma_rejects_unsupported_contracts():
    from flag_gems.runtime.backend._arm.fused.gemma_norm import prepare_gemma_triton

    ops = prepare_gemma_triton()
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    w = torch.zeros(128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="epsilon"):
        ops.gemma_rmsnorm.triton(x, w, 1e-5)
    with pytest.raises(RuntimeError, match="BF16"):
        ops.gemma_rmsnorm.triton(x.float(), w, 1e-6)
