import pytest
import torch
from qwen_image_cpu.bf16_sme import ops as bf16_ops
from qwen_image_cpu.qkv_mixed import ops
from qwen_image_cpu.w8a8_sme import ops as w8_ops
from qwen_image_cpu.w8a8_sme import quantize_weight


@pytest.mark.parametrize(
    "shape", [(0, 35, 32), (1, 35, 32), (37, 129, 72), (128, 256, 512)]
)
@torch.inference_mode()
def test_joint_qkv_is_exact_to_sequential_frozen_arithmetic(shape):
    torch.set_num_threads(4)
    torch.manual_seed(1001)
    m, n, k = shape
    x = torch.randn(m, k).bfloat16()
    w8, bf16, native = w8_ops(), bf16_ops(), ops()
    packed = []
    gold = []
    for _ in range(2):
        q, s = quantize_weight(torch.randn(n, k).bfloat16())
        packed.append(w8.pack_neon(q, s))
        gold.append(w8.linear_bf16(x, w8.pack(q, s), n, 4, 0, True))
    v = bf16.pack_direct(torch.randn(n, k).bfloat16())
    gold.append(bf16.linear_dynamic(x, v, n, k, 4, 32, 256))
    lhs = bf16.pack_lhs_neon(x)
    a8 = w8.repack_neon_lhs(w8.quant_pack_bf16(x), m, k)
    for workers in [1, 2, 3]:
        actual = native.matmul(
            lhs, a8, *packed, v, m, n, k, 4, workers, 32, 256, 32, 64
        )
        assert all(torch.equal(a, b) for a, b in zip(actual, gold))
