import copy

import pytest
import torch
from qwen_image_cpu.cpu_accumulation import enable_fp32_accumulation
from qwen_image_cpu.torch_baseline import (
    FP32Linear,
    configure_linear_workers,
    promote_linears,
)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bias", [True, False])
def test_parallel_linear_keeps_all_rows_and_channels(axis, bias):
    torch.manual_seed(51)
    layer = FP32Linear(torch.nn.Linear(256, 1031, bias=bias).bfloat16())
    x = torch.randn(2, 137, 256).bfloat16()
    expected = layer(x)
    try:
        configure_linear_workers(3, axis)
        actual = layer(x)
    finally:
        configure_linear_workers()
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_promoted_linear_matches_fp32_accumulation(bias, dtype):
    torch.manual_seed(17)
    layer = torch.nn.Linear(64, 96, bias=bias).bfloat16()
    x = torch.randn(2, 7, 64).to(dtype)
    expected = torch.nn.functional.linear(
        x.float(),
        layer.weight.float(),
        layer.bias.float() if bias else None,
    ).to(dtype)
    actual = FP32Linear(layer)(x)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_promoted_transformer_preserves_bf16_boundaries(monkeypatch):
    from qwen_image_cpu import reference

    monkeypatch.setattr(
        reference.transformer_source, "dispatch_attention_fn", reference._cpu_dispatch
    )
    torch.manual_seed(45)
    model = (
        reference.QwenImage21Transformer2DModel(
            in_channels=64,
            out_channels=64,
            num_layers=2,
            attention_head_dim=64,
            num_attention_heads=2,
            context_in_dim=64,
            axes_dims_rope=(16, 24, 24),
        )
        .bfloat16()
        .eval()
    )
    promoted = copy.deepcopy(model)
    count = enable_fp32_accumulation(model)
    assert promote_linears(promoted) == count
    assert not any(isinstance(m, torch.nn.Linear) for m in promoted.modules())
    kwargs = dict(
        hidden_states=torch.randn(1, 16, 64).bfloat16(),
        encoder_hidden_states=torch.randn(1, 8, 64).bfloat16(),
        timestep=torch.tensor([0.8]).bfloat16(),
        img_shapes=[[(1, 4, 4)]],
        img_mask=torch.cat(
            [torch.zeros(1, 8, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)],
            dim=1,
        ),
        encoder_hidden_states_mask=torch.ones(1, 8, dtype=torch.bool),
        return_dict=False,
    )
    for cache_mode in (None, "extract", "cached"):
        if cache_mode == "extract":
            a_cache = reference.transformer_source.QwenImage21KVCache(2)
            b_cache = reference.transformer_source.QwenImage21KVCache(2)
        extra_a = (
            {}
            if cache_mode is None
            else dict(kv_cache=a_cache, kv_cache_mode=cache_mode)
        )
        extra_b = (
            {}
            if cache_mode is None
            else dict(kv_cache=b_cache, kv_cache_mode=cache_mode)
        )
        expected = model(**kwargs, **extra_a)[0]
        actual = promoted(**kwargs, **extra_b)[0]
        assert actual.dtype == expected.dtype == torch.bfloat16
        assert torch.equal(actual, expected)
