import pytest
import torch
from qwen_image_cpu.pointwise_fusion import enable_swiglu, ops, silu_table
from torch.nn import functional as F


@pytest.mark.parametrize("shape", [(0, 17), (1, 1), (3, 7), (129, 97), (1024, 12288)])
@torch.inference_mode()
def test_swiglu_random_and_noncontiguous(shape):
    torch.set_num_threads(4)
    torch.manual_seed(315)
    m, k = shape
    gate = torch.randn(k, m).bfloat16().t()
    up = torch.randn(k, m).bfloat16().t()
    assert torch.equal(ops().swiglu(gate, up, silu_table()), F.silu(gate) * up)


@torch.inference_mode()
def test_all_encodings_and_subnormal_infinity_nan_behavior():
    torch.set_num_threads(4)
    gate = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    for value in [0.0, 1.0, -1.0, 1.5, 1e-20, 1e20]:
        up = torch.full_like(gate, value)
        expected = F.silu(gate) * up
        actual = ops().swiglu(gate, up, silu_table())
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        valid = ~torch.isnan(expected)
        assert torch.equal(
            actual[valid].view(torch.int16), expected[valid].view(torch.int16)
        )


@torch.inference_mode()
def test_adapter_preserves_model_parameters_and_matches_reference():
    from qwen_image_cpu.cpu_accumulation import enable_fp32_accumulation
    from qwen_image_cpu.reference import transformer_source

    torch.set_num_threads(4)
    torch.manual_seed(316)
    model = transformer_source.QwenImage21SwiGLUFeedForward(64, 192).bfloat16().eval()
    enable_fp32_accumulation(model)
    x = torch.randn(1, 129, 64).bfloat16()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    expected = model(x)
    assert enable_swiglu(model) == 1 and enable_swiglu(model) == 0
    assert torch.equal(model(x), expected)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize(
    "m,k", [(0, 1), (1, 1), (3, 7), (4, 8), (31, 65), (35, 129), (97, 96), (129, 12288)]
)
@torch.inference_mode()
def test_fused_pack_matches_library_bytes(m, k):
    from qwen_image_cpu.bf16_sme import ops as bf16_ops
    from qwen_image_cpu.bf16_variants import ops as variant_ops

    torch.set_num_threads(4)
    torch.manual_seed(318)
    gate = torch.randn(k, m).bfloat16().t()
    up = torch.randn(k, m).bfloat16().t()
    expected = bf16_ops().pack_lhs_reference(F.silu(gate) * up)
    actual = ops().swiglu_pack(gate, up, silu_table(), variant_ops().packed_rows())
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_packed_adapter_and_weight_cache_invalidation():
    from qwen_image_cpu.bf16_sme import enable_bf16_sme
    from qwen_image_cpu.cpu_accumulation import enable_fp32_accumulation
    from qwen_image_cpu.reference import transformer_source

    torch.set_num_threads(4)
    torch.manual_seed(319)
    model = transformer_source.QwenImage21SwiGLUFeedForward(64, 192).bfloat16().eval()
    enable_fp32_accumulation(model)
    enable_bf16_sme(model, workers=4, axis=0, dynamic=True)
    x = torch.randn(1, 129, 64).bfloat16()
    expected = model(x)
    assert enable_swiglu(model, packed=True) == 1
    assert torch.equal(model(x), expected)
    model.out.weight.mul_(0.5)
    actual = model(x)
    model._fused_swiglu_enabled = False
    assert torch.equal(actual, model(x))
