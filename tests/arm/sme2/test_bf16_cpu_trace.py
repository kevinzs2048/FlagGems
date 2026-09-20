import pytest
import torch

from flag_gems.runtime.backend._arm.quantized_linear.sme2.bf16_sme import (
    ops as base_ops,
)
from flag_gems.runtime.backend._arm.quantized_linear.sme2.bf16_variants import ops


@pytest.mark.parametrize(
    "shape,order",
    [((0, 35, 64), 0), ((35, 67, 72), 0), ((35, 67, 72), 2), ((128, 256, 512), 1)],
)
def test_trace_covers_each_tile_without_changing_output(shape, order):
    torch.set_num_threads(18)
    torch.manual_seed(992)
    m, n, k = shape
    native = ops()
    base = base_ops()
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    lhs, rhs = base.pack_lhs_reference(x), base.pack_direct(w)
    expected = native.matmul(lhs, rhs, m, n, k, 18, 32, 64, 2, order)
    actual, trace = native.matmul_trace(lhs, rhs, m, n, k, 18, 32, 64, 2, order)
    assert torch.equal(actual, expected)
    assert trace.dtype == torch.long and trace.shape == (
        ((m + 31) // 32) * ((n + 63) // 64),
        8,
    )
    assert {(r, c) for r, c in trace[:, 2:4].tolist()} == {
        (r, c) for r in range(0, m, 32) for c in range(0, n, 64)
    }
    assert (trace[:, 7] >= trace[:, 6]).all()
    assert (trace[:, 0] == 0).all()
