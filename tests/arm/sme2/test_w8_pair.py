import pytest
import torch

from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
    ops,
    quantize_weight,
)


@pytest.mark.parametrize(
    "m,n,k", [(0, 97, 96), (1, 97, 96), (129, 257, 128), (65, 257, 12288)]
)
@torch.inference_mode()
def test_two_projections_share_quantization_without_changing_outputs(m, n, k):
    torch.set_num_threads(4)
    torch.manual_seed(306)
    a = ops()
    a.configure_parallel(False, 4)
    x = torch.randn(k, m).bfloat16().t()
    weights = [a.pack(*quantize_weight(torch.randn(n, k).bfloat16())) for _ in range(2)]
    expected = [a.linear_bf16(x, w, n, 4, 0, True) for w in weights]
    actual = a.linear_pair_bf16(x, *weights, n, 4)
    assert all(torch.equal(v, e) for v, e in zip(actual, expected))
