import pytest
import torch

from flag_gems.runtime.backend._arm.quantized_linear.sme2.bf16_sme import (
    enable_bf16_sme,
    ops,
)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (0, 7, 1),
        (1, 7, 1),
        (3, 35, 7),
        (4, 32, 8),
        (31, 65, 9),
        (32, 65, 64),
        (35, 97, 129),
        (97, 97, 96),
    ],
)
@torch.inference_mode()
def test_neon_pack_tails_noncontiguous_and_gemm(m, n, k):
    torch.set_num_threads(4)
    torch.manual_seed(309)
    native = ops()
    x = torch.randn(k, m).bfloat16().t()
    assert torch.equal(native.pack_lhs_reference(x), native.pack_lhs_neon(x))
    packed = native.pack_direct(torch.randn(n, k).bfloat16())
    assert torch.equal(
        native.linear_dynamic(x, packed, n, k, 4, 32, 256),
        native.linear_dynamic_neon_pack(x, packed, n, k, 4, 32, 256),
    )


@torch.inference_mode()
def test_neon_pack_all_finite_bf16_encodings():
    torch.set_num_threads(4)
    value = (
        torch.arange(-32768, 32768, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(256, 256)
    )
    value[~torch.isfinite(value)] = 0
    assert torch.equal(ops().pack_lhs_reference(value), ops().pack_lhs_neon(value))


@torch.inference_mode()
def test_module_neon_pack_preserves_bias_and_output_dtype():
    torch.set_num_threads(4)
    torch.manual_seed(311)
    layer = torch.nn.Linear(97, 257, bias=True, dtype=torch.bfloat16)
    x = torch.randn(1, 129, 97).bfloat16()
    enable_bf16_sme(layer, workers=4, axis=0, dynamic=True)
    expected = layer(x)
    layer._bf16_sme_neon_pack = True
    actual = layer(x)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)
