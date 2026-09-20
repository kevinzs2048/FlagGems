import pytest
import torch
from qwen_image_cpu.bf16_sme import ops as bf16_ops
from qwen_image_cpu.bf16_variants import ops as variant_ops
from qwen_image_cpu.pointwise_fusion import silu_table
from qwen_image_cpu.w8_swiglu import ops
from qwen_image_cpu.w8a8_sme import ops as w8_ops
from qwen_image_cpu.w8a8_sme import quantize_weight


@pytest.mark.parametrize(
    "shape",
    [
        (0, 97, 96),
        (1, 1, 8),
        (7, 13, 32),
        (37, 257, 96),
        (129, 513, 256),
        (1024, 512, 4096),
    ],
)
@pytest.mark.parametrize("tile", [(32, 128), (64, 256)])
@torch.inference_mode()
def test_packed_output_matches_independent_silu_and_packer(shape, tile):
    torch.set_num_threads(4)
    torch.manual_seed(1601)
    m, n, k = shape
    w8 = w8_ops()
    bf16 = bf16_ops()
    x = torch.randn(k, m).bfloat16().t()
    weights = [
        w8.pack(*quantize_weight(torch.randn(n, k).bfloat16())) for _ in range(2)
    ]
    gate, up = [w8.linear_bf16(x, w, n, 4, 0, True).bfloat16() for w in weights]
    expected = bf16.pack_lhs_reference(torch.nn.functional.silu(gate) * up)
    actual = ops().run(w8.quant_pack_bf16(x), *weights, silu_table(), m, n, k, 4, *tile)
    assert torch.equal(actual, expected)
    # Validate the downstream consumer, not just bytes in a produced buffer.
    down = torch.randn(17, n).bfloat16()
    rhs = bf16.pack_direct(down)
    matrix = variant_ops()
    a = matrix.matmul(actual, rhs, m, 17, n, 4, 32, 256, 2, 0)
    b = matrix.matmul(expected, rhs, m, 17, n, 4, 32, 256, 2, 0)
    assert torch.equal(a, b)


@torch.inference_mode()
def test_model_adapter_and_accuracy_diagnostic_fallbacks():
    from qwen_image_cpu.bf16_sme import enable_bf16_sme
    from qwen_image_cpu.pointwise_fusion import enable_swiglu
    from qwen_image_cpu.reference import transformer_source
    from qwen_image_cpu.w8_swiglu import enable, select
    from qwen_image_cpu.w8a8_sme import DynamicW8SMELinear

    torch.set_num_threads(4)
    torch.manual_seed(1603)
    layer = transformer_source.QwenImage21SwiGLUFeedForward(128, 256).bfloat16().eval()
    for name in ["gate_layer", "proj"]:
        old = getattr(layer, name)
        q, s = quantize_weight(old.weight)
        replacement = DynamicW8SMELinear(q, s)
        replacement.dynamic_tiles = True
        replacement.direct_a8_pack = True
        setattr(layer, name, replacement)
    enable_bf16_sme(layer, workers=4, axis=0, dynamic=True)
    enable_swiglu(layer, packed=True)
    assert enable(layer) == 1 and enable(layer) == 0
    for m, a8 in [(137, True), (9, True), (137, False)]:
        layer.gate_layer.activation_quantization = (
            layer.proj.activation_quantization
        ) = a8
        x = torch.randn(1, m, 128).bfloat16()
        select(layer, False)
        expected = layer(x)
        select(layer, True)
        actual = layer(x)
        assert torch.equal(actual, expected)
