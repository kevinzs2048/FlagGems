import pytest
import torch
from qwen_image_cpu.bf16_register_epilogue import ops
from qwen_image_cpu.bf16_sme import ops as base_ops
from qwen_image_cpu.bf16_variants import ops as variant_ops


@pytest.mark.parametrize(
    "m,n,k",
    [
        (0, 17, 33),
        (1, 1, 1),
        (3, 17, 33),
        (37, 257, 65),
        (128, 512, 256),
        (1024, 512, 4096),
    ],
)
@torch.inference_mode()
def test_register_epilogue_matches_full_fp32_output_rounding(m, n, k):
    torch.set_num_threads(4)
    torch.manual_seed(943)
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    lhs = base_ops().pack_lhs_reference(x)
    rhs = base_ops().pack_direct(w)
    expected = variant_ops().matmul(lhs, rhs, m, n, k, 4, 32, 256, 2, 0).bfloat16()
    actual = ops().matmul(lhs, rhs, m, n, k, 4, 32, 256)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_register_epilogue_preserves_ties_and_special_values():
    torch.set_num_threads(4)
    x = torch.ones(3, 2, dtype=torch.bfloat16)
    pairs = [
        [1.0, 2**-8],
        [1.0 + 2**-7, 2**-8],
        [-1.0, -(2**-8)],
        [-1.0 - 2**-7, -(2**-8)],
        [float("inf"), 0.0],
        [float("-inf"), 0.0],
        [float("nan"), 0.0],
        [2**-126, 2**-133],
        [0.0, -0.0],
    ]
    w = torch.tensor(pairs, dtype=torch.bfloat16)
    lhs = base_ops().pack_lhs_reference(x)
    rhs = base_ops().pack_direct(w)
    expected = (
        variant_ops().matmul(lhs, rhs, 3, len(pairs), 2, 4, 32, 256, 2, 0).bfloat16()
    )
    actual = ops().matmul(lhs, rhs, 3, len(pairs), 2, 4, 32, 256)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("which", ["lhs", "rhs", "tile", "workers"])
@torch.inference_mode()
def test_register_epilogue_rejects_invalid_packed_inputs(which):
    torch.set_num_threads(4)
    lhs = base_ops().pack_lhs_reference(torch.zeros(3, 33, dtype=torch.bfloat16))
    rhs = base_ops().pack_direct(torch.zeros(17, 33, dtype=torch.bfloat16))
    workers, mt = 4, 32
    if which == "lhs":
        lhs = lhs[:-1]
    if which == "rhs":
        rhs = rhs[:-1]
    if which == "tile":
        mt = 3
    if which == "workers":
        workers = 0
    with pytest.raises(RuntimeError):
        ops().matmul(lhs, rhs, 3, 17, 33, workers, mt, 256)


@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_linear_dispatch_and_fallback(bias):
    from qwen_image_cpu.bf16_register_epilogue import enable, select
    from qwen_image_cpu.bf16_sme import enable_bf16_sme

    torch.set_num_threads(4)
    torch.manual_seed(1750)
    layer = torch.nn.Linear(128, 256, bias=bias).bfloat16().eval()
    enable_bf16_sme(layer, workers=4, axis=0, dynamic=True, neon_pack=True)
    assert enable(layer) == (0 if bias else 1)
    for m in [3, 137]:
        x = torch.randn(1, m, 128).bfloat16()
        select(layer, False)
        expected = layer(x)
        select(layer, True)
        actual = layer(x)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_packed_mlp_dispatch(quantized):
    from qwen_image_cpu.bf16_register_epilogue import enable, select
    from qwen_image_cpu.bf16_sme import enable_bf16_sme
    from qwen_image_cpu.pointwise_fusion import enable_swiglu
    from qwen_image_cpu.reference import transformer_source
    from qwen_image_cpu.w8_swiglu import enable as enable_w8
    from qwen_image_cpu.w8a8_sme import DynamicW8SMELinear, quantize_weight

    torch.set_num_threads(4)
    torch.manual_seed(1751)
    layer = transformer_source.QwenImage21SwiGLUFeedForward(128, 256).bfloat16().eval()
    if quantized:
        for name in ["gate_layer", "proj"]:
            q, s = quantize_weight(getattr(layer, name).weight)
            replacement = DynamicW8SMELinear(q, s)
            replacement.dynamic_tiles = True
            replacement.direct_a8_pack = True
            setattr(layer, name, replacement)
    enable_bf16_sme(layer, workers=4, axis=0, dynamic=True, neon_pack=True)
    enable_swiglu(layer, packed=True)
    if quantized:
        assert enable_w8(layer) == 1
    assert enable(layer) > 0
    x = torch.randn(1, 137, 128).bfloat16()
    select(layer, False)
    expected = layer(x)
    select(layer, True)
    actual = layer(x)
    assert torch.equal(actual, expected)
