import pytest
import torch
from qwen_image_cpu.rmsnorm_fusion import enable_rmsnorm, ops


def reference(x, w, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).bfloat16() * w


def exact_with_nan(actual, expected):
    assert torch.equal(actual.isnan(), expected.isnan())
    valid = ~expected.isnan()
    assert torch.equal(
        actual[valid].view(torch.int16), expected[valid].view(torch.int16)
    )


@pytest.mark.parametrize(
    "shape",
    [
        (0, 128),
        (1, 128),
        (7, 128),
        (1, 37, 32, 128),
        (1, 1024, 32, 128),
        (2, 129, 4, 128),
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.inference_mode()
def test_rmsnorm_shape_rounding_and_noncontiguous(shape, eps):
    torch.set_num_threads(4)
    torch.manual_seed(932)
    storage = torch.randn(*shape, 2).bfloat16()
    x = storage[..., 0]
    w = torch.randn(128).bfloat16()
    exact_with_nan(ops().rmsnorm128(x, w, eps), reference(x, w, eps))


@torch.inference_mode()
def test_rmsnorm_all_bf16_encodings_and_extreme_rows():
    torch.set_num_threads(4)
    torch.manual_seed(933)
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    w = torch.randn(128).bfloat16()
    for x in [
        values.reshape(512, 128),
        values[:, None].expand(-1, 128),
        torch.zeros(3, 128, dtype=torch.bfloat16),
    ]:
        exact_with_nan(ops().rmsnorm128(x, w, 1e-6), reference(x, w, 1e-6))


def test_rmsnorm_adapter_toggle_and_grad_fallback():
    from qwen_image_cpu.reference import transformer_source

    torch.set_num_threads(4)
    attention = transformer_source.QwenImage21Attention(128, 1, 128).bfloat16()
    x = torch.randn(1, 7, 1, 128).bfloat16()
    with torch.inference_mode():
        expected = attention.norm_q(x)
        assert enable_rmsnorm(attention) == 2
        assert enable_rmsnorm(attention) == 0
        exact_with_nan(attention.norm_q(x), expected)
        attention.norm_q._fused_rmsnorm_enabled = False
        exact_with_nan(attention.norm_q(x), expected)
        attention.norm_q._fused_rmsnorm_enabled = True
    train_x = x.detach().requires_grad_(True)
    attention.norm_q(train_x).float().sum().backward()
    assert train_x.grad is not None and train_x.grad.isfinite().all()
    with torch.inference_mode():
        fp32 = x.float()
        exact_with_nan(
            attention.norm_q(fp32), attention.norm_q._fused_rmsnorm_original(fp32)
        )
