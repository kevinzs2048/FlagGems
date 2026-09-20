import pytest
import torch
from qwen_image_cpu.bf16_sme import enable_bf16_sme, ops


@torch.inference_mode()
def test_direct_pack_all_finite_bf16_encodings():
    torch.set_num_threads(4)
    value = (
        torch.arange(-32768, 32768, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(256, 256)
    )
    value[~torch.isfinite(value)] = 0
    native = ops()
    assert torch.equal(native.pack(value), native.pack_direct(value))
    assert torch.equal(native.pack_lhs_reference(value), native.pack_lhs_direct(value))


@pytest.mark.parametrize("shape", [(1, 1), (31, 7), (32, 64), (65, 129), (97, 96)])
@torch.inference_mode()
def test_direct_pack_matches_library_bytes_and_gemm(shape):
    torch.set_num_threads(4)
    torch.manual_seed(58)
    n, k = shape
    # Noncontiguous input also exercises the canonical layout boundary.
    w = torch.randn(k, n).bfloat16().t()
    native = ops()
    reference = native.pack(w)
    direct = native.pack_direct(w)
    assert torch.equal(reference, direct)
    x = torch.randn(35, k).bfloat16()
    assert torch.equal(
        native.linear(x, reference, n, k, 4, 0), native.linear(x, direct, n, k, 4, 0)
    )


@pytest.mark.parametrize(
    "shape", [(0, 1), (1, 7), (31, 64), (32, 64), (65, 129), (97, 96)]
)
@torch.inference_mode()
def test_direct_lhs_pack_and_gemm(shape):
    torch.set_num_threads(4)
    torch.manual_seed(60)
    m, k = shape
    x = torch.randn(k, m).bfloat16().t()
    native = ops()
    assert torch.equal(native.pack_lhs_reference(x), native.pack_lhs_direct(x))
    w = torch.randn(97, k).bfloat16()
    packed = native.pack_direct(w)
    for axis in (0, 1):
        assert torch.equal(
            native.linear(x, packed, 97, k, 4, axis),
            native.linear_direct(x, packed, 97, k, 4, axis),
        )


@pytest.mark.parametrize(
    "shape", [(0, 68, 97), (1, 68, 97), (17, 68, 96), (35, 65, 129)]
)
@pytest.mark.parametrize("axis", [0, 1])
@torch.inference_mode()
def test_sme_bf16_tails_against_fp32(shape, axis):
    torch.set_num_threads(4)
    torch.manual_seed(46)
    m, n, k = shape
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(n, k).bfloat16()
    native = ops()
    actual = native.linear(x, native.pack(w), n, k, 4, axis)
    expected = torch.nn.functional.linear(x.float(), w.float())
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)


@torch.no_grad()
def test_sme_bf16_module_bias_and_weights_unchanged():
    torch.manual_seed(47)
    module = torch.nn.Linear(97, 68, bias=True, dtype=torch.bfloat16)
    x = torch.randn(1, 17, 97).bfloat16()
    state = {k: v.clone() for k, v in module.state_dict().items()}
    expected = (
        torch.nn.functional.linear(x.float(), module.weight.float(), None)
        + module.bias.float()
    )
    assert enable_bf16_sme(module, workers=4, axis=0) == 1
    assert enable_bf16_sme(module) == 0
    actual = module(x)
    torch.testing.assert_close(
        actual.float(), expected.bfloat16().float(), atol=0.00390625, rtol=0.004
    )
    assert actual.dtype == torch.bfloat16
    assert all(torch.equal(v, state[k]) for k, v in module.state_dict().items())
    module.weight.mul_(0.5)
    updated = module(x)
    expected = (
        torch.nn.functional.linear(x.float(), module.weight.float(), None)
        + module.bias.float()
    )
    torch.testing.assert_close(
        updated.float(), expected.bfloat16().float(), atol=0.00390625, rtol=0.004
    )


@pytest.mark.parametrize("m,nn,k", [(0, 1, 7), (1, 1, 7), (17, 13, 96), (35, 33, 129)])
@torch.inference_mode()
def test_hybrid_bf16_distinct_layouts_and_tails(m, nn, k):
    torch.set_num_threads(6)
    torch.manual_seed(113)
    x = torch.randn(m, k).bfloat16()
    w = torch.randn(32 + nn, k).bfloat16()
    native = ops()
    actual = native.linear_hybrid(
        x, native.pack_direct(w[:32]), native.pack_neon(w[32:]), 32, nn, 2, 4
    )
    expected = torch.nn.functional.linear(x.double(), w.double()).float()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


@torch.inference_mode()
def test_logical_pool_rank_dispatch():
    torch.set_num_threads(4)
    torch.manual_seed(114)
    native = ops()
    x = torch.randn(65, 97).bfloat16()
    w = torch.randn(97, 97).bfloat16()
    p = native.pack_direct(w)
    expected = native.linear(x, p, 97, 97, 4, 0)
    for ranks in [[0, 3], [1, 2], [0], [3]]:
        assert torch.equal(native.linear_pool_ranks(x, p, 97, 97, ranks, 0), expected)
    for ranks in [[0, 0], [-1], [4], []]:
        with pytest.raises(RuntimeError):
            native.linear_pool_ranks(x, p, 97, 97, ranks, 0)
