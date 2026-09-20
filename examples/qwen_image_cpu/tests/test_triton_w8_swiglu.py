import pytest
import torch
from qwen_image_cpu.bf16_sme import ops as bf16_ops
from qwen_image_cpu.bf16_variants import ops as matrix_ops
from qwen_image_cpu.pointwise_fusion import silu_table
from qwen_image_cpu.triton_w8_linear import pack_weight
from qwen_image_cpu.w8a8_sme import ops, quantize_weight

from flag_gems.runtime.backend._arm.ops.w8a8_packed_sme2 import w8a8_packed_sme2_bf16
from flag_gems.runtime.backend._arm.ops.w8a8_swiglu_packed_sme2 import (
    fused_swiglu_packed,
)


@pytest.mark.parametrize(
    "m,n,k", [(0, 64, 256), (1, 64, 256), (19, 128, 256), (64, 64, 4096)]
)
@pytest.mark.parametrize("column_major", [False, True])
@torch.inference_mode()
def test_projection_pair_silu_and_downstream(m, n, k, column_major):
    torch.set_num_threads(4)
    torch.manual_seed(1731)
    w8, bf16 = ops(), bf16_ops()
    x = torch.randn(m, k).bfloat16()
    weights = [quantize_weight(torch.randn(n, k).bfloat16()) for _ in range(2)]
    native = [w8.pack(*w) for w in weights]
    packed = [pack_weight(*w) for w in weights]
    g, u = [w8.linear_bf16(x, w, n, 4, 0, True).bfloat16() for w in native]
    expected = bf16.pack_lhs_reference(torch.nn.functional.silu(g) * u)
    actual = fused_swiglu_packed(
        w8.quant_pack_bf16(x),
        *packed,
        silu_table(),
        m,
        packed_rows=matrix_ops().packed_rows(),
        workers=4,
        chunk=4,
        column_major=column_major,
    )
    assert torch.equal(actual, expected)
    rhs = bf16.pack_direct(torch.randn(17, n).bfloat16())
    a = matrix_ops().matmul(actual, rhs, m, 17, n, 4, 32, 256, 2, 0)
    b = matrix_ops().matmul(expected, rhs, m, 17, n, 4, 32, 256, 2, 0)
    assert torch.equal(a, b)


@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("hardware_bf16", [False, True])
@pytest.mark.parametrize(
    "shape", [(0, 64, 256), (1, 64, 256), (37, 128, 256), (1024, 64, 4096)]
)
@torch.inference_mode()
def test_direct_bf16_output(shape, hardware_bf16, column_major):
    torch.set_num_threads(4)
    torch.manual_seed(1733)
    m, n, k = shape
    x = torch.randn(m, k).bfloat16()
    codes, scales = quantize_weight(torch.randn(n, k).bfloat16())
    w = pack_weight(codes, scales)
    lhs = ops().quant_pack_bf16(x)
    expected = ops().matmul(lhs, ops().pack(codes, scales), m, n, k, 4, 0).bfloat16()
    got = w8a8_packed_sme2_bf16(
        lhs,
        w.data,
        w.sums,
        w.scales,
        m,
        n,
        k,
        workers=4,
        hardware_bf16=hardware_bf16,
        column_major=column_major,
    )
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", [0, 1, 37, 128])
@torch.inference_mode()
def test_paired_direct_bf16_output(m):
    torch.set_num_threads(4)
    torch.manual_seed(1734)
    n, k = 128, 256
    x = torch.randn(m, k).bfloat16()
    weights = [quantize_weight(torch.randn(n, k).bfloat16()) for _ in range(2)]
    w = pack_weight(
        torch.cat([a for a, b in weights]), torch.cat([b for a, b in weights])
    )
    lhs = ops().quant_pack_bf16(x)
    expected = torch.stack(
        [ops().matmul(lhs, ops().pack(*p), m, n, k, 4, 0).bfloat16() for p in weights]
    )
    actual = w8a8_packed_sme2_bf16(
        lhs, w.data, w.sums, w.scales, m, 2 * n, k, workers=4, pair_n=n
    )
    assert actual.shape == (2, m, n) and actual.is_contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
