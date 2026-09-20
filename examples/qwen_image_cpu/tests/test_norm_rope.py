import pytest
import torch
from qwen_image_cpu.norm_rope import ops


def reference(x, w, f, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normalized = (x * torch.rsqrt(variance + eps)).bfloat16() * w
    complex_x = torch.view_as_complex(normalized.float().reshape(*x.shape[:-1], 64, 2))
    return torch.view_as_real(complex_x * f.unsqueeze(1)).flatten(3).bfloat16()


def exact(a, b):
    assert torch.equal(a.isnan(), b.isnan())
    valid = ~b.isnan()
    assert torch.equal(a[valid].view(torch.int16), b[valid].view(torch.int16))


@pytest.mark.parametrize(
    "shape",
    [
        (0, 7, 2, 128),
        (1, 0, 2, 128),
        (2, 7, 3, 128),
        (1, 37, 32, 128),
        (1, 1024, 32, 128),
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.inference_mode()
def test_rounding_batch_noncontiguous_and_empty(shape, eps):
    torch.set_num_threads(4)
    torch.manual_seed(1510)
    b, s, h, d = shape
    x = torch.randn(b, h, s, d).bfloat16().transpose(1, 2)
    w = torch.randn(256).bfloat16()[::2]
    angle = torch.randn(64, s).t()
    f = torch.polar(torch.ones_like(angle), angle)
    exact(ops().run(x, w, f, eps), reference(x, w, f, eps))


@torch.inference_mode()
def test_all_bf16_encodings_and_extreme_rows():
    torch.set_num_threads(4)
    torch.manual_seed(1511)
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    w = torch.randn(128).bfloat16()
    for x in [
        values.reshape(1, 512, 1, 128),
        values[None, :, None, None].expand(1, -1, 1, 128),
        torch.zeros(1, 3, 1, 128, dtype=torch.bfloat16),
    ]:
        angle = torch.randn(x.shape[1], 64)
        f = torch.polar(torch.ones_like(angle), angle)
        exact(ops().run(x, w, f, 1e-6), reference(x, w, f, 1e-6))


@pytest.mark.parametrize("cached", [False, True])
@torch.inference_mode()
def test_adapter_preserves_reference_cache_and_disabled_fallback(cached, monkeypatch):
    from qwen_image_cpu.norm_rope import _rotated, enable, select
    from qwen_image_cpu.reference import transformer_source as source

    torch.set_num_threads(4)
    torch.manual_seed(1513)
    original = source._qwenimage21_prepare_qkv
    monkeypatch.setattr(source, "_qwenimage21_prepare_qkv", original)
    monkeypatch.setattr(source, "apply_rotary_emb_qwen", source.apply_rotary_emb_qwen)
    attn = source.QwenImage21Attention(256, 2, 128).bfloat16().eval()
    x = torch.randn(1, 137, 256).bfloat16()
    angle = torch.randn(137, 64)
    freq = torch.polar(torch.ones_like(angle), angle)
    base_cache, cache = (
        source.QwenImage21KVLayerCache(),
        source.QwenImage21KVLayerCache(),
    )
    expected = original(attn, x, freq, base_cache, "extract", slice(0, 9))
    assert enable(attn) == 1 and enable(attn) == 0
    actual = source._qwenimage21_prepare_qkv(
        attn, x, freq, cache, "extract", slice(0, 9)
    )
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
    assert all(torch.equal(a, b) for a, b in zip(cache.get(), base_cache.get()))
    if cached:
        x = x[:, 9:]
        freq = freq[9:]
        select(attn, False)
        expected = source._qwenimage21_prepare_qkv(
            attn, x, freq, base_cache, "cached", None
        )
        select(attn, True)
        actual = source._qwenimage21_prepare_qkv(attn, x, freq, cache, "cached", None)
        assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
    for rotary in [None, freq]:
        select(attn, False)
        expected = source._qwenimage21_prepare_qkv(attn, x, rotary, None, None, None)
        select(attn, True)
        actual = source._qwenimage21_prepare_qkv(attn, x, rotary, None, None, None)
        assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
    assert _rotated.get() is None


@torch.inference_mode()
def test_exception_resets_context(monkeypatch):
    from qwen_image_cpu.norm_rope import _rotated, enable
    from qwen_image_cpu.reference import transformer_source as source

    torch.set_num_threads(4)
    monkeypatch.setattr(
        source, "_qwenimage21_prepare_qkv", source._qwenimage21_prepare_qkv
    )
    monkeypatch.setattr(source, "apply_rotary_emb_qwen", source.apply_rotary_emb_qwen)
    attn = source.QwenImage21Attention(128, 1, 128).bfloat16().eval()
    enable(attn)

    def fail(*args, **kwargs):
        raise RuntimeError("injected cache failure")

    cache = source.QwenImage21KVLayerCache()
    monkeypatch.setattr(cache, "store", fail)
    with pytest.raises(RuntimeError, match="injected cache failure"):
        source._qwenimage21_prepare_qkv(
            attn,
            torch.randn(1, 7, 128).bfloat16(),
            torch.ones(7, 64, dtype=torch.complex64),
            cache,
            "extract",
            slice(0, 2),
        )
    assert _rotated.get() is None
