import pytest
import torch
from qwen_image_cpu.bf16_sme import ops as base_ops
from qwen_image_cpu.bf16_tile_epilogue import ops
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
def test_tile_epilogue_matches_full_fp32_output_rounding(m, n, k):
    torch.set_num_threads(4)
    torch.manual_seed(943)
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    lhs = base_ops().pack_lhs_reference(x)
    rhs = base_ops().pack_direct(w)
    expected = variant_ops().matmul(lhs, rhs, m, n, k, 4, 32, 256, 2, 0).bfloat16()
    actual = ops().matmul(lhs, rhs, m, n, k, 4, 32, 256)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
