import os

import pytest
import torch

from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
    ops,
    quantize_weight,
)


@pytest.mark.parametrize(
    "m,n,k,split",
    [(0, 128, 32, 64), (1, 129, 32, 64), (37, 257, 264, 192), (128, 512, 4096, 320)],
)
@torch.inference_mode()
def test_cooperative_jobs_and_trace_preserve_complete_outputs(m, n, k, split):
    torch.set_num_threads(4)
    torch.manual_seed(977)
    native = ops()
    x = torch.randn(m, k).bfloat16()
    q, s = quantize_weight(torch.randn(n, k).bfloat16())
    full = native.pack(q, s)
    ps = native.pack(q[:split], s[:split])
    pn = native.pack_neon(q[split:], s[split:])
    expected = native.linear_bf16(x, full, n, 4, 0, True)
    for mapping in [
        torch.zeros(os.cpu_count(), dtype=torch.long),
        torch.arange(os.cpu_count()) // 6,
    ]:
        args = (x, ps, pn, split, n - split, 4, mapping, 64, 256, 32, 64)
        assert torch.equal(native.linear_hybrid_cooperative(*args), expected)
        actual, trace = native.linear_hybrid_cooperative_trace(*args)
        assert torch.equal(actual, expected)
        total = ((m + 63) // 64) * ((split + 255) // 256) + ((m + 31) // 32) * (
            (n - split + 63) // 64
        )
        assert trace.shape == (total, 8)
        assert len({(r[0], r[2], r[3]) for r in trace.tolist()}) == total
        if m:
            assert (trace[:, 1] >= 0).all() and (trace[:, 1] < 4).all()
            assert (trace[:, 7] >= trace[:, 6]).all()


@torch.inference_mode()
def test_cooperative_adapter_policy_switch():
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.hybrid_dynamic import (
        configure_cooperative_hybrid,
        configure_dynamic_hybrid,
        enable_dynamic_hybrid,
    )
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
        DynamicW8SMELinear,
    )

    torch.set_num_threads(18)
    torch.manual_seed(978)
    q, s = quantize_weight(torch.randn(256, 128).bfloat16())
    x = torch.randn(128, 128).bfloat16()
    layer = DynamicW8SMELinear(q, s)
    expected = layer(x)
    enable_dynamic_hybrid(layer)
    configure_cooperative_hybrid(layer, 12, 0.75, torch.arange(os.cpu_count()) // 6)
    assert torch.equal(layer(x), expected)
    layer._dynamic_hybrid_trace_enabled = True
    assert torch.equal(layer(x), expected)
    assert layer._dynamic_hybrid_last_trace.shape[1] == 8
    layer._dynamic_hybrid_trace_enabled = False
    configure_dynamic_hybrid(layer, 3, 15, 0.5)
    assert not layer._dynamic_hybrid_cooperative
    assert torch.equal(layer(x), expected)
