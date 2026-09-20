import pytest
import torch
from qwen_image_cpu import reference
from qwen_image_cpu.bf16_sme import enable_bf16_sme
from qwen_image_cpu.qkv_mixed import (
    _projection_outputs,
    enable_joint_qkv,
    prepack_joint_qkv,
    select_joint_qkv,
)
from qwen_image_cpu.w8a8_sme import DynamicW8SMELinear, quantize_weight


def attention():
    attn = (
        reference.transformer_source.QwenImage21Attention(128, 2, 64).bfloat16().eval()
    )
    for name in ["to_q", "to_k"]:
        layer = getattr(attn, name)
        q, s = quantize_weight(layer.weight)
        setattr(attn, name, DynamicW8SMELinear(q, s))
    enable_bf16_sme(attn, workers=4, axis=0, dynamic=True)
    return attn


@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("backend", ["mixed", "paired_sme"])
@torch.inference_mode()
def test_reference_qkv_norm_rope_and_prefix_cache_are_preserved(
    rotary, backend, monkeypatch
):
    torch.set_num_threads(4)
    torch.manual_seed(1004)
    source = reference.transformer_source
    original = source._qwenimage21_prepare_qkv
    monkeypatch.setattr(source, "_qwenimage21_prepare_qkv", original)
    attn = attention()
    x = torch.randn(1, 137, 128).bfloat16()
    frequencies = (
        torch.polar(torch.ones(137, 32), torch.randn(137, 32)) if rotary else None
    )
    baseline_cache, candidate_cache = (
        source.QwenImage21KVLayerCache(),
        source.QwenImage21KVLayerCache(),
    )
    expected = original(attn, x, frequencies, baseline_cache, "extract", slice(0, 9))
    assert enable_joint_qkv(attn, backend) == 1 and enable_joint_qkv(attn, backend) == 0
    assert prepack_joint_qkv(attn) == 1
    actual = source._qwenimage21_prepare_qkv(
        attn, x, frequencies, candidate_cache, "extract", slice(0, 9)
    )
    assert (
        all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
        and actual[3] == expected[3]
    )
    assert all(
        torch.equal(a, b) for a, b in zip(candidate_cache.get(), baseline_cache.get())
    )
    assert _projection_outputs.get() is None
    tail = x[:, 9:]
    freq = frequencies[9:] if rotary else None
    select_joint_qkv(attn, False)
    expected = source._qwenimage21_prepare_qkv(
        attn, tail, freq, baseline_cache, "cached", None
    )
    select_joint_qkv(attn, True)
    actual = source._qwenimage21_prepare_qkv(
        attn, tail, freq, candidate_cache, "cached", None
    )
    assert (
        all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
        and actual[3] == expected[3]
    )
    assert _projection_outputs.get() is None


@torch.inference_mode()
def test_small_m_and_disabled_activation_quantization_fall_back(monkeypatch):
    torch.set_num_threads(4)
    torch.manual_seed(1005)
    source = reference.transformer_source
    original = source._qwenimage21_prepare_qkv
    monkeypatch.setattr(source, "_qwenimage21_prepare_qkv", original)
    attn = attention()
    assert enable_joint_qkv(attn) == 1
    for length, quantized in [(9, True), (128, False)]:
        attn.to_q.activation_quantization = attn.to_k.activation_quantization = (
            quantized
        )
        x = torch.randn(1, length, 128).bfloat16()
        expected = original(attn, x, None, None, None, None)
        actual = source._qwenimage21_prepare_qkv(attn, x, None, None, None, None)
        assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
        assert _projection_outputs.get() is None


@torch.inference_mode()
def test_reference_exception_does_not_leave_projection_context(monkeypatch):
    torch.set_num_threads(4)
    torch.manual_seed(1006)
    source = reference.transformer_source
    monkeypatch.setattr(
        source, "_qwenimage21_prepare_qkv", source._qwenimage21_prepare_qkv
    )
    attn = attention()
    enable_joint_qkv(attn)

    def fail(*args, **kwargs):
        raise RuntimeError("injected norm failure")

    monkeypatch.setattr(attn.norm_q, "forward", fail)
    with pytest.raises(RuntimeError, match="injected norm failure"):
        source._qwenimage21_prepare_qkv(
            attn, torch.randn(1, 128, 128).bfloat16(), None, None, None, None
        )
    assert _projection_outputs.get() is None
