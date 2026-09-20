import pytest
import torch
from qwen_image_cpu.w8a8_sme import ops, quantize_weight


@pytest.mark.parametrize(
    "m,n,k,split",
    [
        (0, 128, 32, 64),
        (1, 129, 32, 64),
        (37, 257, 264, 192),
        (128, 512, 4096, 320),
        (1024, 512, 4096, 256),
    ],
)
@torch.inference_mode()
def test_dynamic_hybrid_preserves_codes_and_full_k_outputs(m, n, k, split):
    torch.set_num_threads(4)
    torch.manual_seed(953)
    x = torch.randn(m, k).bfloat16()
    q, s = quantize_weight(torch.randn(n, k).bfloat16())
    native = ops()
    full = native.pack(q, s)
    ps = native.pack(q[:split], s[:split])
    pn = native.pack_neon(q[split:], s[split:])
    expected = native.linear_bf16(x, full, n, 4, 0, True)
    for sw, nw in [(1, 3), (2, 2), (3, 1)]:
        actual = native.linear_hybrid_dynamic(
            x, ps, pn, split, n - split, sw, nw, 64, 256, 32, 64
        )
        assert torch.equal(actual, expected)


def test_adapter_cached_modes_and_normal_tensor_weight_updates():
    from qwen_image_cpu.hybrid_dynamic import (
        configure_dynamic_hybrid,
        enable_dynamic_hybrid,
        prepack_dynamic_hybrid,
        select_dynamic_hybrid,
    )
    from qwen_image_cpu.w8a8_sme import DynamicW8SMELinear

    torch.set_num_threads(18)
    torch.manual_seed(955)
    q, s = quantize_weight(torch.randn(256, 128).bfloat16())
    layer = DynamicW8SMELinear(q, s, torch.randn(256).bfloat16())
    layer.direct_a8_pack = True
    layer.dynamic_tiles = True
    x = torch.randn(128, 128).bfloat16()
    with torch.inference_mode():
        expected = layer(x)
        assert enable_dynamic_hybrid(layer) == 1
        assert enable_dynamic_hybrid(layer) == 0
        assert prepack_dynamic_hybrid(layer) == 1
        assert torch.equal(layer(x), expected)
        for sw, nw, fraction in [(3, 6, 0.75), (6, 6, 0.75), (6, 12, 0.5)]:
            configure_dynamic_hybrid(layer, sw, nw, fraction)
            assert torch.equal(layer(x), expected)
        assert len(layer._dynamic_hybrid_weights) == 2
        select_dynamic_hybrid(layer, False)
        assert torch.equal(layer(x), expected)
        select_dynamic_hybrid(layer, True)
        for short in [x[:7], x.float()]:
            assert torch.equal(layer(short), layer._dynamic_hybrid_original(short))
        key = layer._dynamic_hybrid_weight_key
        layer.weight_scale.mul_(0.5)
        fresh = ops().pack(layer.weight_codes, layer.weight_scale)
        expected = (
            ops().linear_bf16(x, fresh, 256, 18, 0, True) + layer.bias.float()
        ).bfloat16()
        assert torch.equal(layer(x), expected)
        assert layer._dynamic_hybrid_weight_key != key


@torch.inference_mode()
def test_adapter_loads_inference_tensors_without_version_counter():
    from qwen_image_cpu.hybrid_dynamic import enable_dynamic_hybrid
    from qwen_image_cpu.w8a8_sme import DynamicW8SMELinear

    torch.set_num_threads(18)
    torch.manual_seed(956)
    q, s = quantize_weight(torch.randn(256, 128).bfloat16())
    assert q.is_inference() and s.is_inference()
    layer = DynamicW8SMELinear(q, s)
    x = torch.randn(128, 128).bfloat16()
    expected = layer(x)
    enable_dynamic_hybrid(layer)
    assert torch.equal(layer(x), expected)


@pytest.mark.parametrize("m", [0, 37, 128])
@torch.inference_mode()
def test_cpu_trace_covers_each_tile_without_changing_output(m):
    import os

    torch.set_num_threads(4)
    torch.manual_seed(957)
    q, s = quantize_weight(torch.randn(512, 256).bfloat16())
    x = torch.randn(m, 256).bfloat16()
    native = ops()
    ps = native.pack(q[:256], s[:256])
    pn = native.pack_neon(q[256:], s[256:])
    args = (x, ps, pn, 256, 256, 2, 2, 64, 256, 32, 64)
    expected = native.linear_hybrid_dynamic(*args)
    actual, trace = native.linear_hybrid_dynamic_trace(*args)
    assert torch.equal(actual, expected)
    assert trace.shape == ((m + 63) // 64 + ((m + 31) // 32) * 4, 8)
    if m:
        assert len({tuple(row[:4]) for row in trace.tolist()}) == len(trace)
        assert (trace[:, 4:6] >= 0).all() and (trace[:, 4:6] < os.cpu_count()).all()
        assert (trace[:, 7] >= trace[:, 6]).all()
        assert (
            (trace[trace[:, 0] == 0, 1] >= 0) & (trace[trace[:, 0] == 0, 1] < 2)
        ).all()
        assert (
            (trace[trace[:, 0] == 1, 1] >= 2) & (trace[trace[:, 0] == 1, 1] < 4)
        ).all()
    numer, denom = native.trace_timebase()
    assert numer > 0 and denom > 0
