import torch
from qwen_image_cpu import reference
from qwen_image_cpu.cpu_accumulation import enable_fp32_accumulation


@torch.inference_mode()
def test_reference_causal_conv_padding_is_preserved():
    torch.manual_seed(43)
    layer = reference.vae_source.QwenImage21CausalConv3d(4, 8, 3, padding=1).bfloat16()
    value = torch.randn(1, 4, 1, 5, 7).bfloat16()
    expected = layer(value)
    assert enable_fp32_accumulation(layer) == 1
    actual = layer(value, None)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_reference_prefix_cache_matches_recomputed_target(monkeypatch):
    torch.manual_seed(44)
    monkeypatch.setattr(
        reference.transformer_source, "dispatch_attention_fn", reference._cpu_dispatch
    )
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
    cache = reference.transformer_source.QwenImage21KVCache(2)
    prefill = model(**kwargs, kv_cache=cache, kv_cache_mode="extract")[0]
    assert all(layer.is_populated for layer in cache.layer_caches)
    kwargs["timestep"] = torch.tensor([0.4]).bfloat16()
    kwargs["hidden_states"] = torch.randn(1, 16, 64).bfloat16()
    expected = model(**kwargs)[0][:, -16:]
    actual = model(**kwargs, kv_cache=cache, kv_cache_mode="cached")[0]
    assert actual.shape == expected.shape == (1, 16, 64)
    assert torch.allclose(actual.float(), expected.float(), atol=0.0078125, rtol=0.004)
    assert prefill.shape == (1, 24, 64)
