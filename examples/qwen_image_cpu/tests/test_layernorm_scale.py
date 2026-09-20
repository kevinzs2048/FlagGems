import pytest
import torch
from qwen_image_cpu.layernorm_scale import ops
from torch.nn import functional as F


@pytest.mark.parametrize(
    "shape,broadcast",
    [
        ((1, 0, 4096), False),
        ((1, 1, 7), False),
        ((2, 5, 129), False),
        ((2, 5, 128), True),
        ((1, 37, 4096), True),
        ((1, 1024, 4096), False),
    ],
)
@torch.inference_mode()
def test_rounding_reduction_broadcast_and_tails(shape, broadcast):
    torch.set_num_threads(4)
    torch.manual_seed(1009)
    x = torch.randn(shape).bfloat16()
    s = torch.randn(shape[0], 1 if broadcast else shape[1], shape[2]).bfloat16()
    expected = F.layer_norm(x, [shape[-1]], eps=1e-6) * s
    actual = ops().run(x, s, 1e-6)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("value", [0.0, 3.0, 100.0, -2.0])
@torch.inference_mode()
def test_constant_rows_match_reference_rounding(value):
    torch.set_num_threads(4)
    x = torch.full((1, 2, 4096), value, dtype=torch.bfloat16)
    s = torch.ones_like(x) * 3
    assert torch.equal(ops().run(x, s, 1e-6), F.layer_norm(x, [4096], eps=1e-6) * s)


@torch.inference_mode()
def test_noncontiguous_and_nonfinite_values():
    torch.set_num_threads(4)
    torch.manual_seed(1010)
    x = torch.randn(2, 129, 7).bfloat16().transpose(1, 2)
    s = torch.randn_like(x)
    x[0, 1, 0] = float("inf")
    x[1, 2, 0] = float("nan")
    expected = F.layer_norm(x, [129], eps=1e-6) * s
    actual = ops().run(x, s, 1e-6)
    assert torch.equal(actual.isnan(), expected.isnan())
    mask = expected.isfinite()
    assert torch.equal(actual.view(torch.int16)[mask], expected.view(torch.int16)[mask])


@pytest.mark.parametrize("masked", [False, True])
@torch.inference_mode()
def test_shared_modulation_adapter_preserves_block_outputs(masked, monkeypatch):
    from qwen_image_cpu import reference
    from qwen_image_cpu.layernorm_scale import enable_normscale
    from qwen_image_cpu.shared_modulation import enable_shared_modulation
    from test_shared_modulation import TinyBlocks

    torch.set_num_threads(4)
    torch.manual_seed(1011)
    monkeypatch.setattr(
        reference.transformer_source, "dispatch_attention_fn", reference._cpu_dispatch
    )
    model = TinyBlocks().bfloat16().eval()
    enable_shared_modulation(model)
    x = torch.randn(2, 7, 16).bfloat16()
    mod = torch.randn(3 if masked else 2, 64).bfloat16()
    mask = torch.arange(7) % 2 == 0 if masked else None
    expected = model(x, mod, mask)
    assert enable_normscale(model) == 2
    assert torch.equal(model(x, mod, mask), expected)
