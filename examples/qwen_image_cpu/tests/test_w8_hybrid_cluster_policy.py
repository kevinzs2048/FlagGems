import os

import pytest
import torch
from qwen_image_cpu.w8a8_sme import ops, quantize_weight


@pytest.mark.parametrize(
    "shape,split", [((0, 129, 32), 64), ((37, 257, 264), 192), ((128, 512, 512), 320)]
)
@torch.inference_mode()
def test_tail_help_and_missing_eligible_cluster_complete_every_tile(shape, split):
    torch.set_num_threads(4)
    torch.manual_seed(996)
    m, n, k = shape
    native = ops()
    x = torch.randn(m, k).bfloat16()
    q, s = quantize_weight(torch.randn(n, k).bfloat16())
    full = native.pack(q, s)
    ps = native.pack(q[:split], s[:split])
    pn = native.pack_neon(q[split:], s[split:])
    expected = native.linear_bf16(x, full, n, 4, 0, True)
    mapping = torch.zeros(os.cpu_count(), dtype=torch.long)
    for mask in [0, 1, 2, 7]:
        args = (x, ps, pn, split, n - split, 4, mapping, mask, 64, 256, 32, 64)
        actual, trace = native.linear_hybrid_policy_trace(*args)
        assert torch.equal(actual, expected)
        assert torch.equal(native.linear_hybrid_policy(*args), expected)
        expected_tiles = {
            (0, r, c) for r in range(0, m, 64) for c in range(0, split, 256)
        }
        expected_tiles |= {
            (1, r, split + c) for r in range(0, m, 32) for c in range(0, n - split, 64)
        }
        assert {(r[0], r[2], r[3]) for r in trace.tolist()} == expected_tiles
        assert trace.shape == (len(expected_tiles), 8)


def test_mask_range_is_checked_before_execution():
    torch.set_num_threads(2)
    native = ops()
    q, s = quantize_weight(torch.ones(128, 32, dtype=torch.bfloat16))
    ps, pn = native.pack(q[:64], s[:64]), native.pack_neon(q[64:], s[64:])
    for mask in [-1, 1 << 32]:
        with pytest.raises(RuntimeError):
            native.linear_hybrid_policy(
                torch.zeros(1, 32, dtype=torch.bfloat16),
                ps,
                pn,
                64,
                64,
                2,
                torch.zeros(os.cpu_count(), dtype=torch.long),
                mask,
                64,
                256,
                32,
                64,
            )


@torch.inference_mode()
def test_adapter_policy_and_trace_restore_to_cooperative():
    from qwen_image_cpu.hybrid_dynamic import (
        configure_cluster_policy,
        configure_cooperative_hybrid,
        enable_dynamic_hybrid,
    )
    from qwen_image_cpu.w8a8_sme import DynamicW8SMELinear

    torch.set_num_threads(18)
    torch.manual_seed(997)
    q, s = quantize_weight(torch.randn(256, 128).bfloat16())
    x = torch.randn(128, 128).bfloat16()
    layer = DynamicW8SMELinear(q, s)
    gold = layer(x)
    enable_dynamic_hybrid(layer)
    mapping = torch.zeros(os.cpu_count(), dtype=torch.long)
    configure_cluster_policy(layer, 18, 0.75, mapping, 0)
    assert torch.equal(layer(x), gold)
    layer._dynamic_hybrid_trace_enabled = True
    assert torch.equal(layer(x), gold)
    assert layer._dynamic_hybrid_last_trace.shape[1] == 8
    configure_cooperative_hybrid(layer, 18, 0.5, mapping)
    assert layer._dynamic_hybrid_neon_mask is None
    assert torch.equal(layer(x), gold)
