import pytest
import torch
from qwen_image_cpu.bf16_sme import ops as base_ops
from qwen_image_cpu.cluster_tiles import ops, read_topology


@pytest.mark.parametrize("shape", [(0, 35, 1), (35, 67, 73), (129, 513, 128)])
@torch.inference_mode()
def test_adaptive_claims_cover_tail_and_cross_row_boundaries(shape):
    torch.set_num_threads(4)
    torch.manual_seed(993)
    m, n, k = shape
    base = base_ops()
    native = ops()
    mapping, _ = read_topology()
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    lhs, rhs = base.pack_lhs_neon(x), base.pack_direct(w)
    expected = base.linear_dynamic(x, rhs, n, k, 4, 32, 256)
    for batches in [[1, 1, 1], [1, 1, 4], [4, 4, 4], [1, 2, 8]]:
        for order in [0, 1]:
            actual = native.matmul(
                lhs, rhs, m, n, k, 4, 32, 32, mapping, torch.tensor(batches), order
            )
            assert torch.equal(actual, expected)


def test_invalid_cpu_policy_is_rejected():
    torch.set_num_threads(2)
    native = ops()
    base = base_ops()
    lhs = base.pack_lhs_neon(torch.zeros(2, 8, dtype=torch.bfloat16))
    rhs = base.pack_direct(torch.zeros(8, 8, dtype=torch.bfloat16))
    for mapping, batches in [([3], [1, 1, 4]), ([-1], [1]), ([0], [0]), ([0], [17])]:
        with pytest.raises(RuntimeError):
            native.matmul(
                lhs,
                rhs,
                2,
                8,
                8,
                2,
                32,
                32,
                torch.tensor(mapping),
                torch.tensor(batches),
                0,
            )
