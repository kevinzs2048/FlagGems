import pytest
import torch

from flag_gems.runtime.backend._arm.quantized_linear.sme2.bf16_sme import (
    ops as bf16_ops,
)
from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import ops as w8_ops
from flag_gems.runtime.backend._arm.quantized_linear.sme2.w8a8_sme import (
    quantize_weight,
)


@pytest.mark.parametrize(
    "torch_threads,kernel_threads", [(1, 3), (2, 6), (4, 2), (1, 18)]
)
@torch.inference_mode()
def test_kernel_owned_teams_match_aten_and_do_not_change_torch_threads(
    torch_threads, kernel_threads
):
    torch.set_num_threads(4)
    torch.manual_seed(301)
    b, a = bf16_ops(), w8_ops()
    x = torch.randn(129, 96).bfloat16()
    w = torch.randn(257, 96).bfloat16()
    b.configure_parallel(False, 4)
    a.configure_parallel(False, 4)
    pb = b.pack_direct(w)
    q, s = quantize_weight(w)
    pa = a.pack(q, s)
    expected_b = b.linear_dynamic(x, pb, 257, 96, 4, 32, 256)
    expected_a = a.linear_bf16(x, pa, 257, 4, 0, True)
    expected_pack = a.quant_pack_bf16(x)
    try:
        torch.set_num_threads(torch_threads)
        for op in [b, a]:
            op.configure_parallel(True, kernel_threads)
        assert torch.equal(b.pack_direct(w), pb)
        assert torch.equal(a.pack(q, s), pa)
        assert torch.equal(a.quant_pack_bf16(x), expected_pack)
        assert torch.equal(
            b.linear_dynamic(x, pb, 257, 96, kernel_threads, 32, 256), expected_b
        )
        assert torch.equal(
            a.linear_bf16(x, pa, 257, kernel_threads, 0, True), expected_a
        )
        assert torch.get_num_threads() == torch_threads
        assert b.parallel_info()[:3] == [1, kernel_threads, torch_threads]
    finally:
        for op in [b, a]:
            op.configure_parallel(False, 18)
        torch.set_num_threads(4)
