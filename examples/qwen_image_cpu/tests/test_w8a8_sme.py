import pytest
import torch
from qwen_image_cpu.w8a8_sme import (
    DynamicW8SMELinear,
    activation_reference,
    ops,
    quantize_weight,
    unpack_activation,
)


@pytest.mark.parametrize(
    "m,n,k", [(0, 17, 32), (1, 7, 8), (17, 65, 96), (33, 97, 128), (17, 68, 12288)]
)
@pytest.mark.parametrize("axis", [0, 1])
@torch.inference_mode()
def test_same_quantized_inputs_against_integer_oracle(m, n, k, axis):
    torch.set_num_threads(4)
    torch.manual_seed(75)
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    codes, scales = quantize_weight(w)
    native = ops()
    lhs = native.quant_pack(x)
    rhs = native.pack(codes, scales)
    actual = native.matmul(lhs, rhs, m, n, k, 4, axis)
    if m:
        a, sa, z = unpack_activation(lhs, m, k)
        ar, sar, zr = activation_reference(x)
        assert torch.equal(a, ar) and torch.equal(z, zr)
        torch.testing.assert_close(sa, sar, rtol=1e-6, atol=0)
        integer = (a.int() - z[:, None]) @ codes.int().T
        expected = integer.float() * sa[:, None] * scales[None, :]
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-5)
    else:
        assert actual.shape == (0, n)


@pytest.mark.parametrize("constant", [0.0, 3.0, -7.0])
@torch.inference_mode()
def test_constant_activation_and_zero_weight(constant):
    torch.set_num_threads(4)
    x = torch.full((17, 32), constant, dtype=torch.bfloat16)
    w = torch.zeros(65, 32, dtype=torch.bfloat16)
    codes, scales = quantize_weight(w)
    assert torch.isfinite(scales).all() and (scales > 0).all()
    out = ops().linear(x, ops().pack(codes, scales), 65, 4, 0)
    assert torch.equal(out, torch.zeros_like(out))
    torch.manual_seed(116)
    codes, scales = quantize_weight(torch.randn(65, 32).bfloat16())
    actual = ops().linear(x, ops().pack(codes, scales), 65, 4, 0)
    a, sa, z = activation_reference(x)
    expected = (
        ((a.int() - z[:, None]) @ codes.int().T).float() * sa[:, None] * scales[None, :]
    )
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-5)


@torch.inference_mode()
def test_module_shape_bias_dtype():
    torch.set_num_threads(4)
    torch.manual_seed(76)
    x = torch.randn(2, 17, 96).bfloat16()
    w = torch.randn(65, 96).bfloat16()
    codes, scales = quantize_weight(w)
    bias = torch.randn(65).bfloat16()
    layer = DynamicW8SMELinear(codes, scales, bias)
    actual = layer(x)
    expected = (
        (ops().linear(x.reshape(-1, 96), layer.prepack(), 65, 4, 0) + bias.float())
        .bfloat16()
        .reshape(2, 17, 65)
    )
    assert torch.equal(actual, expected) and actual.dtype == torch.bfloat16


def test_reject_nonfinite_weight_and_invalid_k():
    with pytest.raises(ValueError):
        quantize_weight(torch.full((16, 32), float("nan"), dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):
        ops().quant_pack(torch.randn(17, 7))


@pytest.mark.parametrize("m,nn,k", [(1, 1, 8), (17, 9, 96), (33, 65, 128)])
@torch.inference_mode()
def test_hybrid_tail_layout_and_same_integer_oracle(m, nn, k):
    torch.set_num_threads(6)
    torch.manual_seed(111)
    ns = 64
    n = ns + nn
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    codes, scales = quantize_weight(w)
    native = ops()
    ps = native.pack(codes[:ns], scales[:ns])
    pn = native.pack_neon(codes[ns:], scales[ns:])
    actual = native.linear_hybrid(x, ps, pn, ns, nn, 2, 4)
    a, sa, z = activation_reference(x)
    expected = (
        ((a.int() - z[:, None]) @ codes.int().T).float() * sa[:, None] * scales[None, :]
    )
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-5)
    assert torch.equal(actual, native.linear(x, native.pack(codes, scales), n, 6, 0))
    with pytest.raises(RuntimeError):
        native.linear_hybrid(x, ps, ps, ns, nn, 2, 4)


@pytest.mark.parametrize("m,k", [(0, 32), (1, 8), (17, 96), (33, 128), (37, 12288)])
@torch.inference_mode()
def test_activation_reorder_preserves_codes_and_metadata(m, k):
    torch.set_num_threads(4)
    torch.manual_seed(115)
    x = torch.randn(m, k).bfloat16()
    native = ops()
    assert torch.equal(
        native.repack_neon_lhs(native.quant_pack(x), m, k), native.quant_pack_neon(x)
    )
