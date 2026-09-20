import pytest
import torch

from qwen_image_cpu.validate_w8 import check_integer_oracle, metrics
from qwen_image_cpu.w8a8_sme import DynamicW8SMELinear, quantize_weight


@pytest.mark.parametrize("k", [96, 12288])
def test_validation_oracle_uses_full_reduction_dimension(k):
    torch.set_num_threads(18)
    torch.manual_seed(92)
    w = torch.randn(129, k).bfloat16()
    codes, scales = quantize_weight(w)
    layer = DynamicW8SMELinear(codes, scales)
    result = check_integer_oracle(layer, torch.randn(2, k).bfloat16())
    assert result["relative_l2"] < 1e-6


def test_metrics_reject_nonfinite_predictions():
    with pytest.raises(ValueError):
        metrics(torch.tensor([float("nan")]), torch.ones(1))
