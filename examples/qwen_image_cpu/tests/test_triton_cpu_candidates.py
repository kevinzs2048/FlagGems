import pytest
import torch
from qwen_image_cpu.pointwise_fusion import ops
from qwen_image_cpu.triton_cpu_candidates import (
    integer_gemm,
    integer_gemm_packed,
    pack_integer_panels,
    residual,
)


@pytest.mark.parametrize(
    "shape,gate_tokens",
    [((1, 9, 128), 1), ((2, 5, 131), 5), ((2, 3, 4096), 1), ((1, 0, 7), 1)],
)
def test_residual_matches_native(shape, gate_tokens):
    torch.manual_seed(812)
    x, y = [torch.randn(shape).bfloat16() for _ in range(2)]
    g = torch.randn(shape[0], gate_tokens, shape[-1]).bfloat16()
    out = residual(x, y, g, workers=3)
    gold = ops().residual(x, y, g)
    assert torch.equal(out.view(torch.int16), gold.view(torch.int16))
    assert torch.equal(out, x + y * g)


def test_residual_noncontiguous():
    torch.manual_seed(9)
    x, y, g = [torch.randn(2, 131, 7).bfloat16().transpose(1, 2) for _ in range(3)]
    assert torch.equal(residual(x, y, g), ops().residual(x, y, g))


def test_residual_special_values():
    # All BF16 encodings include signs, subnormals, infinities and NaN payloads.
    x = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 512, 128)
    )
    y = x.flip(1)
    g = torch.linspace(-2, 2, 128).bfloat16().reshape(1, 1, 128)
    got = residual(x, y, g)
    expected = ops().residual(x, y, g)
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "m,n,k", [(16, 16, 128), (19, 21, 257), (32, 32, 4096), (0, 16, 128)]
)
def test_integer_gemm_full_k_and_tails(m, n, k):
    torch.manual_seed(33)
    a = torch.randint(-128, 128, (m, k), dtype=torch.int8)
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8)
    got = integer_gemm(a, w, workers=3)
    # Independent wide-integer reference, not another quantized backend.
    expected = a.to(torch.int64) @ w.to(torch.int64).T
    assert torch.equal(got.to(torch.int64), expected)


def test_integer_gemm_bound():
    a = torch.full((16, 32768), -128, dtype=torch.int8)
    out = integer_gemm(a, a)
    assert torch.equal(out, torch.full_like(out, 536870912))


@pytest.mark.parametrize("k", [128, 4096])
@pytest.mark.parametrize("tile", [16, 32])
def test_packed_integer_layout_and_dot(k, tile):
    torch.manual_seed(945)
    a, w = [torch.randint(-128, 128, (32, k), dtype=torch.int8) for _ in range(2)]
    pa, pw = pack_integer_panels(a), pack_integer_panels(w)
    assert torch.equal(pa.permute(0, 3, 1, 2, 4).reshape_as(a), a)
    assert torch.equal(pw.permute(0, 3, 1, 2, 4).reshape_as(w), w)
    got = integer_gemm_packed(pa, pw, workers=3, tile=tile)
    assert torch.equal(got.to(torch.int64), a.to(torch.int64) @ w.to(torch.int64).T)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        residual(torch.ones(3), torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError):
        integer_gemm(torch.ones(4, 4), torch.ones(4, 4))
    with pytest.raises(ValueError):
        integer_gemm(
            torch.ones(4, 8, dtype=torch.int8).T, torch.ones(4, 4, dtype=torch.int8)
        )
