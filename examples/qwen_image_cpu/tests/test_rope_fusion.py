import pytest
import torch
from qwen_image_cpu.rope_fusion import enable_rope, ops


def reference(x, freqs):
    value = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
    return torch.view_as_real(value * freqs.unsqueeze(1)).flatten(3).to(x.dtype)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 7, 3, 2),
        (1, 0, 3, 6),
        (1, 3, 1, 2),
        (2, 7, 3, 6),
        (1, 37, 32, 128),
        (1, 1024, 32, 128),
    ],
)
@torch.inference_mode()
def test_rope_tail_batch_noncontiguous_and_rounding(shape):
    torch.set_num_threads(4)
    torch.manual_seed(323)
    b, s, h, d = shape
    x = torch.randn(b, h, s, d).bfloat16().transpose(1, 2)
    angle = torch.randn(d // 2, s).t()
    freqs = torch.polar(torch.ones_like(angle), angle)
    actual = ops().rope(x, freqs, 2)
    expected = reference(x, freqs)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_rope_all_bf16_encodings_and_special_values():
    torch.set_num_threads(4)
    torch.manual_seed(324)
    x = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 1024, 1, 64)
    )
    angle = torch.randn(1024, 32)
    for freqs in [
        torch.polar(torch.ones_like(angle), angle),
        torch.ones_like(angle, dtype=torch.complex64),
    ]:
        expected = reference(x, freqs)
        actual = ops().rope(x, freqs, 2)
        assert torch.equal(actual.isnan(), expected.isnan())
        valid = ~expected.isnan()
        assert torch.equal(
            actual[valid].view(torch.int16), expected[valid].view(torch.int16)
        )


@torch.inference_mode()
def test_rope_adapter_toggle_and_real_path_fallback():
    from qwen_image_cpu.reference import transformer_source

    torch.set_num_threads(4)
    original = transformer_source.apply_rotary_emb_qwen
    x = torch.randn(1, 7, 3, 8).bfloat16()
    angle = torch.randn(7, 4)
    freqs = torch.polar(torch.ones_like(angle), angle)
    expected = original(x, freqs, use_real=False)
    try:
        assert enable_rope() and not enable_rope()
        patched = transformer_source.apply_rotary_emb_qwen
        assert torch.equal(patched(x, freqs, use_real=False), expected)
        patched.enabled = False
        assert torch.equal(patched(x, freqs, use_real=False), expected)
        patched.enabled = True
        real = torch.randn(1, 3, 7, 8).bfloat16()
        cosine = torch.randn(7, 8)
        sine = torch.randn(7, 8)
        assert torch.equal(
            patched(real, (cosine, sine), use_real=True),
            original(real, (cosine, sine), use_real=True),
        )
    finally:
        transformer_source.apply_rotary_emb_qwen = original
