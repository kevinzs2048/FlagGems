import pytest
import torch
from qwen_image_cpu.bf16_sme import ops as bf16_ops
from qwen_image_cpu.w8a8_sme import ops as w8_ops
from qwen_image_cpu.w8a8_sme import quantize_weight


@pytest.mark.parametrize(
    "m,n,k", [(0, 97, 96), (1, 97, 96), (65, 129, 96), (97, 67, 128)]
)
@pytest.mark.parametrize("workers", [1, 2, 4])
@torch.inference_mode()
def test_dynamic_partition_is_bitwise_equal_including_tails(m, n, k, workers):
    torch.set_num_threads(4)
    torch.manual_seed(202)
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    b = bf16_ops()
    packed = b.pack_direct(w)
    baseline = b.linear(x, packed, n, k, 4, 0)
    assert torch.equal(b.linear_dynamic(x, packed, n, k, workers, 32, 64), baseline)
    q, s = quantize_weight(w)
    a = w8_ops()
    packed = a.pack(q, s)
    baseline = a.linear(x, packed, n, 4, 0)
    for dynamic in [False, True]:
        assert torch.equal(
            a.linear_tiled(x, packed, n, workers, 32, 64, dynamic), baseline
        )


@torch.inference_mode()
def test_dynamic_bf16_odd_k_and_invalid_tile():
    torch.set_num_threads(4)
    torch.manual_seed(203)
    x = torch.randn(65, 129).bfloat16()
    w = torch.randn(97, 129).bfloat16()
    b = bf16_ops()
    packed = b.pack_direct(w)
    assert torch.equal(
        b.linear_dynamic(x, packed, 97, 129, 4, 32, 64),
        b.linear(x, packed, 97, 129, 4, 0),
    )
    for mt, nt in [(0, 64), (32, 0), (7, 64), (32, 7)]:
        with pytest.raises(RuntimeError):
            b.linear_dynamic(x, packed, 97, 129, 4, mt, nt)
