import pytest
import torch

from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
    ops,
    quantize_weight,
)


@pytest.mark.parametrize(
    "m,k", [(0, 32), (1, 8), (17, 96), (33, 128), (37, 4096), (65, 12288)]
)
@torch.inference_mode()
def test_bf16_pack_matches_independent_f32_packer(m, k):
    torch.set_num_threads(4)
    torch.manual_seed(204)
    # Noncontiguous source and several magnitudes, positive/negative/constant rows.
    x = torch.randn(k, m).bfloat16().t()
    for exponent in [-20, 0, 20]:
        value = x * (2.0**exponent)
        if m > 3:
            value[0] = 0
            value[1] = 3
            value[2] = -7
        native = ops()
        assert torch.equal(native.quant_pack_bf16(value), native.quant_pack(value))
        w = torch.randn(97, k).bfloat16()
        q, s = quantize_weight(w)
        packed = native.pack(q, s)
        for dynamic in [False, True]:
            actual = native.linear_bf16(value, packed, 97, 4, 0, dynamic)
            assert torch.equal(actual, native.linear(value, packed, 97, 4, 0))


@torch.inference_mode()
def test_bf16_pack_codes_near_rounding_ties_and_subnormal_values():
    torch.set_num_threads(4)
    # This includes signed zero/subnormals and tie-sensitive BF16 magnitudes.
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values[~values.isfinite()] = 0
    # Avoid overflow of the row-wide high-low range in the original quantizer.
    values = values.clamp(-1e30, 1e30).reshape(512, 128)
    native = ops()
    assert torch.equal(native.quant_pack_bf16(values), native.quant_pack(values))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dynamic", [False, True])
@torch.inference_mode()
def test_direct_pack_module_preserves_bias_dtype_and_dynamic_tails(dtype, dynamic):
    from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
        DynamicW8SMELinear,
    )

    torch.set_num_threads(4)
    torch.manual_seed(208)
    q, s = quantize_weight(torch.randn(257, 96).bfloat16())
    module = DynamicW8SMELinear(q, s, torch.randn(257).bfloat16())
    x = torch.randn(96, 129, dtype=dtype).t()
    expected = module(x)
    module.dynamic_tiles = dynamic
    module.direct_a8_pack = True
    actual = module(x)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
