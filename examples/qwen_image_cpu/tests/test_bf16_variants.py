import pytest
import torch
from qwen_image_cpu.bf16_sme import ops as base_ops
from qwen_image_cpu.bf16_variants import ops


@pytest.mark.parametrize("m,n,k", [(0, 7, 1), (1, 7, 1), (35, 97, 129), (129, 513, 96)])
@torch.inference_mode()
def test_microkernel_and_task_order_preserve_accumulation(m, n, k):
    torch.set_num_threads(4)
    torch.manual_seed(312)
    b = base_ops()
    x = torch.randn(m, k).bfloat16()
    lhs = b.pack_lhs_neon(x)
    rhs = b.pack_direct(torch.randn(n, k).bfloat16())
    expected = b.linear_dynamic(x, rhs, n, k, 4, 32, 256)
    for version in [1, 2]:
        for order in range(4):
            assert torch.equal(
                ops().matmul(lhs, rhs, m, n, k, 4, 32, 256, version, order), expected
            )


def test_arithmetic_probe_streaming_entry_and_operation_count():
    torch.set_num_threads(2)
    native = ops()
    a = native.arithmetic_probe(1, 16, False)
    assert a > 0 and native.arithmetic_probe(2, 8, False) == a
    assert native.arithmetic_probe(1, 16, True) == 2 * a
    for workers, iterations in [(0, 16), (3, 16), (1, 0), (1, 100000001)]:
        with pytest.raises(RuntimeError):
            native.arithmetic_probe(workers, iterations, False)
