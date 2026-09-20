import pytest
import torch
from qwen_image_cpu import triton_w8_linear as candidate
from qwen_image_cpu.w8a8_sme import ops, quantize_weight


@pytest.mark.parametrize("row_epilogue", [False, True])
@pytest.mark.parametrize("tile_n", [16, 32, 64])
@pytest.mark.parametrize(
    "m,n,k,chunk",
    [
        (1, 64, 256, 1),
        (19, 128, 256, 4),
        (32, 64, 4096, 2),
        (37, 64, 256, 8),
        (0, 64, 256, 1),
    ],
)
def test_complete_w8_linear_and_native_input_abi(m, n, k, chunk, tile_n, row_epilogue):
    torch.manual_seed(118)
    x = torch.randn(m, k).bfloat16()
    codes, scales = quantize_weight(torch.randn(n, k).bfloat16())
    weight = candidate.pack_weight(codes, scales)
    native = ops()
    lhs = native.quant_pack_bf16(x)
    rhs = native.pack(codes, scales)
    actual = candidate.from_packed_activation(
        lhs, weight, m, workers=3, chunk=chunk, tile_n=tile_n, row_epilogue=row_epilogue
    )
    expected = native.matmul(lhs, rhs, m, n, k, 3, 0)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    bias = torch.randn(n).bfloat16()
    got = candidate.linear(
        x,
        weight,
        bias,
        workers=3,
        chunk=chunk,
        tile_n=tile_n,
        row_epilogue=row_epilogue,
    )
    assert torch.equal(
        got.view(torch.int16), (expected + bias.float()).bfloat16().view(torch.int16)
    )


def test_invalid_weight_and_input():
    with pytest.raises(ValueError):
        candidate.pack_weight(torch.zeros(63, 256, dtype=torch.int8), torch.ones(63))
    with pytest.raises(ValueError):
        candidate.pack_weight(torch.zeros(64, 256, dtype=torch.int8), torch.zeros(64))
    weight = candidate.pack_weight(
        torch.zeros(64, 256, dtype=torch.int8), torch.ones(64)
    )
    with pytest.raises(ValueError):
        candidate.from_packed_activation(torch.empty(17, dtype=torch.uint8), weight, 1)
    with pytest.raises(ValueError):
        candidate.linear(torch.ones(1, 256), weight)


def test_zero_and_constant_activation():
    torch.manual_seed(44)
    codes = torch.randint(-127, 128, (64, 256), dtype=torch.int8)
    scales = torch.rand(64) * 0.1 + 0.001
    weight = candidate.pack_weight(codes, scales)
    x = torch.stack([torch.zeros(256), torch.ones(256), -torch.ones(256)]).bfloat16()
    expected = (
        ops().linear_bf16(x, ops().pack(codes, scales), 64, 3, 0, True).bfloat16()
    )
    assert torch.equal(candidate.linear(x, weight, workers=3), expected)
