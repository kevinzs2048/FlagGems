# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional Arm native/TLE operators, independent numerical and state oracles."""

import inspect
import os
import platform
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in {"arm64", "aarch64"},
    reason="requires AArch64",
)


@pytest.fixture(scope="module")
def native():
    from flag_gems.runtime.backend._arm.fused import arm_native_ops

    backend = (
        Path(__file__).resolve().parents[1] / "src/flag_gems/runtime/backend/_arm/lib"
    )
    if not os.environ.get("FLAGGEMS_ARM_NATIVE_LIBRARY") and not list(
        backend.glob("libflag_gems_arm_native.*")
    ):
        pytest.skip("build FLAGGEMS_BUILD_ARM_KAI=ON")
    return arm_native_ops


@pytest.fixture
def raw(native):
    ops = pytest.importorskip("triton.language.extra.cpu.tle_ops")
    if "result_dtype" not in inspect.signature(ops.raw_call).parameters:
        pytest.skip("requires CPU scalar native-call support with void results")
    native("gdn", raw=True)
    return native


def _a8(x):
    low = torch.minimum(x.amin(-1), torch.zeros(x.shape[0]))
    high = torch.maximum(x.amax(-1), torch.zeros(x.shape[0]))
    multiplier = torch.where(high == low, 1.0, 255.0 / (high - low))
    use_min = -128.0 + low * multiplier + 127.0 + high * multiplier > 0
    zp = (
        torch.where(use_min, -128.0 - low * multiplier, 127.0 - high * multiplier)
        .clamp(-128, 127)
        .round()
    )
    scaled = x * multiplier[:, None]
    code = (scaled.sign() * (scaled.abs() + 0.5).floor() + zp[:, None]).clamp(-128, 127)
    return code - zp[:, None], 1.0 / multiplier


@pytest.mark.parametrize("rows", [1, 3, 16, 127, 512])
@pytest.mark.parametrize("n,split", [(128, 0), (128, 64), (128, 128), (640, 0)])
def test_arm_g32_shared_and_raw(native, raw, rows, n, split):
    torch.manual_seed(276 + rows)
    ops = native("w4a8_g32")
    k = 96
    x = torch.randn(rows, k)
    w = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scale = (0.01 + torch.rand(n, k // 32) * 0.02).bfloat16()
    try:
        packed = ops.pack(w, scale, split)[0]
    except RuntimeError as error:
        if split and "ISA is unavailable" in str(error):
            pytest.skip("SME2 unavailable")
        raise
    q, sx = _a8(x)
    accum = torch.einsum(
        "mgk,ngk->mng", q.reshape(rows, -1, 32), w.float().reshape(n, -1, 32)
    )
    expected = (accum * scale.float()[None]).sum(-1) * sx[:, None]
    args = [x, packed, n, split, 4, 2]
    out = ops.linear(*args)
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)
    assert torch.equal(out, native("w4a8_g32", raw=True).linear(*args))
    assert torch.equal(out, ops.linear_shared(*args))
    assert torch.equal(out, native("w4a8_g32", raw=True).linear_shared(*args))
    canonical = native("w8a8").canonical(x)
    torch.testing.assert_close(ops.linear_a8(canonical, *args[1:]), out, rtol=0, atol=0)
    assert torch.equal(
        out, native("w4a8_g32", raw=True).linear_a8(canonical, *args[1:])
    )


@pytest.mark.parametrize("rows,n,k", [(1, 17, 96), (3, 17, 96), (16, 64, 128)])
@pytest.mark.parametrize("sme", [False, True])
def test_arm_w8_tail_and_quantization(native, raw, rows, n, k, sme):
    torch.manual_seed(976 + rows)
    ops = native("w8a8")
    x = torch.randn(rows, k)
    x[0].zero_()
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scales = torch.rand(n) * 0.02 + 0.001
    try:
        packed = ops.pack(w, scales, sme)
    except RuntimeError as error:
        if sme and "ISA is unavailable" in str(error):
            pytest.skip("SME2 unavailable")
        raise
    q, sx = _a8(x)
    expected = (q @ w.float().t()) * sx[:, None] * scales[None]
    args = [x, packed, n, sme, 32, 128]
    out = ops.linear(*args)
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)
    assert torch.equal(out, native("w8a8", raw=True).linear(*args))


@pytest.mark.parametrize("rows", [0, 1, 7, 16])
@pytest.mark.parametrize("silu", [False, True])
def test_arm_conv_prefill_state_and_empty_ranges(native, rows, silu):
    from flag_gems.runtime.backend._arm.gdn import gdn_conv1d_prefill_triton

    torch.manual_seed(29 + rows)
    dim = 127
    x = torch.randn(rows, dim, dtype=torch.bfloat16).t()
    initial = torch.randn(3, dim, 5, dtype=torch.bfloat16)
    state = initial.clone()
    weight = torch.randn(dim, 4, dtype=torch.bfloat16)
    bias = torch.randn(dim, dtype=torch.bfloat16)
    starts = torch.tensor([0, 0, rows], dtype=torch.int32)
    slots = torch.tensor([2, 0], dtype=torch.int32)
    flags = torch.tensor([False, True])
    expected_state = initial.clone()
    expected_state[2, :, :3].zero_()
    history = initial[0, :, :3].float()
    expected = torch.empty_like(x)
    for token in range(rows):
        value = x[:, token].float()
        result = bias.float() + history[:, 0] * weight[:, 0].float()
        result = result + history[:, 1] * weight[:, 1].float()
        result = result + history[:, 2] * weight[:, 2].float()
        result = result + value * weight[:, 3].float()
        expected[:, token] = torch.nn.functional.silu(result) if silu else result
        history = torch.cat([history[:, 1:], value[:, None]], -1)
    expected_state[0, :, :3] = history
    out = gdn_conv1d_prefill_triton(x, state, weight, bias, starts, slots, flags, silu)
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.016)
    assert torch.equal(state, expected_state)
    # Unselected slots and speculative rollback history must remain unchanged.
    assert torch.equal(state[1], initial[1])
    assert torch.equal(state[:, :, 3:], initial[:, :, 3:])


@pytest.mark.parametrize("rows", [0, 1, 3, 16])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_arm_conv_update_triton_state_and_tail(native, rows, silu, has_bias):
    from flag_gems.runtime.backend._arm.gdn import gdn_conv1d_update_triton

    torch.manual_seed(721 + rows)
    dim = 127
    # Exercise positive noncontiguous input/history/weight strides and tails.
    x = torch.randn(rows, dim * 2, dtype=torch.bfloat16)[:, ::2]
    original = torch.randn(rows + 2, dim, 6, dtype=torch.bfloat16)
    state = original.clone()[:, :, ::2]
    weight = torch.randn(dim, 8, dtype=torch.bfloat16)[:, ::2]
    bias = torch.randn(dim, dtype=torch.bfloat16) if has_bias else None
    slots = torch.arange(rows, 0, -1, dtype=torch.int32)
    expected_state = original[:, :, ::2].clone()
    expected = torch.empty_like(x)
    for token, slot in enumerate(slots):
        history = expected_state[slot].float()
        result = history[:, 0] * weight[:, 0].float()
        if bias is not None:
            result = bias.float() + result
        result = result + history[:, 1] * weight[:, 1].float()
        result = result + history[:, 2] * weight[:, 2].float()
        result = result + x[token].float() * weight[:, 3].float()
        expected[token] = torch.nn.functional.silu(result) if silu else result
        expected_state[slot, :, :2] = expected_state[slot, :, 1:].clone()
        expected_state[slot, :, 2] = x[token]
    untouched = x.clone()
    out = gdn_conv1d_update_triton(x, state, weight, bias, slots, silu)
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.016)
    assert torch.equal(state, expected_state)
    assert torch.equal(x, untouched)
    assert torch.equal(state[0], original[0, :, ::2])
    assert torch.equal(state[-1], original[-1, :, ::2])


@pytest.mark.parametrize("rows", [1, 7, 32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_arm_moe_router_ties_and_fusion(native, raw, monkeypatch, rows, renormalize):
    torch.manual_seed(911 + rows)
    h, inter, experts, top = 128, 64, 8, 3

    def pack(n, k):
        w = torch.randint(-8, 8, (n, k), dtype=torch.int8) + 8
        codes = (w[:, ::2] | (w[:, 1::2] << 4)).byte()
        scale = (0.001 + torch.rand(n, k // 32) * 0.01).bfloat16()
        return torch.ops.aten._dyn_quant_pack_4bit_weight(codes, scale, None, 32, k, n)

    w1 = torch.stack([pack(2 * inter, h) for _ in range(experts)])
    w2 = torch.stack([pack(h, inter) for _ in range(experts)])
    x = torch.randn(rows, h, dtype=torch.bfloat16)
    logits = torch.zeros(rows, experts, dtype=torch.bfloat16)
    values, indices = logits.topk(top, sorted=False)
    gates = (
        values.softmax(-1)
        if renormalize
        else (values - logits.logsumexp(-1, keepdim=True)).exp()
    ).float()
    expected = torch.zeros(rows, h)
    for token in range(rows):
        for route in indices[token].argsort():
            expert = indices[token, route]
            gu = torch.ops.aten._dyn_quant_matmul_4bit(
                x[token : token + 1].float(), w1[expert], 32, h, inter * 2
            )
            act = torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]
            value = torch.ops.aten._dyn_quant_matmul_4bit(act, w2[expert], 32, inter, h)
            expected[token] += value[0] * gates[token, route]
    args = [x, logits, w1, w2, h, inter, top, renormalize]
    ops = native("moe")
    out = ops.moe_fused(*args)
    torch.testing.assert_close(out, expected.bfloat16(), rtol=0.02, atol=0.001)
    assert torch.equal(out, native("moe", raw=True).moe_fused(*args))
    torch.testing.assert_close(out, ops.moe(*args), rtol=0.02, atol=0.001)

    for name in (
        "moe_prefill_routed",
        "moe_prefill_local",
        "moe_prefill_fused",
        "moe_prefill_dynamic",
        "moe_prefill_adaptive",
        "moe_prefill_tiled",
        "moe_prefill_balanced",
    ):
        call_args = [*args, 16] if name == "moe_prefill_tiled" else args
        value = getattr(native("w4a8_g32"), name)(*call_args)
        torch.testing.assert_close(value, expected.bfloat16(), rtol=0.02, atol=0.001)
        assert torch.equal(
            value, getattr(native("w4a8_g32", raw=True), name)(*call_args)
        )
    streamed = ops.streamed(*args, 64)
    torch.testing.assert_close(streamed, expected.bfloat16(), rtol=0.02, atol=0.001)
    assert torch.equal(streamed, native("moe", raw=True).streamed(*args, 64))
    assert torch.equal(ops.moe(*args), ops.moe_blocked(*args))
    if rows == 1:
        monkeypatch.setenv("FLAGGEMS_ARM_MOE_DECODE_THREADS", "2")
        for name in ("moe_decode_routed", "moe_decode_fused"):
            value = getattr(native("w4a8_g32"), name)(*args)
            torch.testing.assert_close(
                value, expected.bfloat16(), rtol=0.02, atol=0.001
            )
            assert torch.equal(
                value, getattr(native("w4a8_g32", raw=True), name)(*args)
            )


def test_arm_raw_errors_do_not_poison_entry(native, raw):
    ops = native("w8a8", raw=True)
    with pytest.raises(RuntimeError, match="Expected contiguous FP32"):
        ops.linear(
            torch.ones(1, 32, dtype=torch.bfloat16),
            torch.empty(1, dtype=torch.uint8),
            8,
            False,
            32,
            128,
        )
    w = torch.zeros(8, 32, dtype=torch.int8)
    scales = torch.ones(8)
    packed = native("w8a8").pack(w, scales, False)
    assert torch.equal(
        ops.linear(torch.ones(1, 32), packed, 8, False, 32, 128), torch.zeros(1, 8)
    )


def test_arm_optional_qwen_tle_fallback():
    import importlib

    module = importlib.import_module(
        "flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp"
    )
    if module._tle_fused_mlp is None:
        assert module.patch_qwen3_mlp(torch.nn.Linear(4, 4)) == 0


def _gdn_step(q, k, v, state, decay, beta, normalize):
    q, k, v = q.float(), k.float(), v.float()
    if normalize:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q / q.shape[-1] ** 0.5
    q = q.repeat_interleave(v.shape[0] // q.shape[0], 0)
    k = k.repeat_interleave(v.shape[0] // k.shape[0], 0)
    decayed = state * decay[:, None, None]
    predicted = (decayed * k[:, None, :]).sum(-1)
    delta = (v - predicted) * beta[:, None]
    updated = decayed + k[:, None, :] * delta[:, :, None]
    return (updated * q[:, None, :]).sum(-1).bfloat16(), updated


@pytest.mark.parametrize("normalize", [False, True])
def test_arm_gdn_recurrence_and_prefill(native, raw, normalize):
    torch.manual_seed(470)
    tokens, kh, vh, kd, vd = 3, 2, 4, 32, 16
    q = torch.randn(1, tokens, kh, kd, dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, tokens, vh, vd, dtype=torch.bfloat16) * 0.1
    original = torch.randn(3, vh, vd, kd) * 0.01
    al = torch.randn(vh) * 0.1
    a = torch.randn(tokens, vh, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    dt = torch.randn(vh, dtype=torch.bfloat16) * 0.1
    g = -al.exp() * torch.nn.functional.softplus(a.float() + dt.float())
    beta = b.float().sigmoid()
    # Decode visits independent slots; prefill visits two sequences, with an
    # empty final sequence. Their references exercise different state ownership.
    indices = torch.tensor([2, 0, 1], dtype=torch.int32)
    expected_state = original.clone()
    expected = torch.empty_like(v)
    for token, slot in enumerate(indices):
        expected[0, token], expected_state[slot] = _gdn_step(
            q[0, token],
            k[0, token],
            v[0, token],
            expected_state[slot],
            g[token].exp(),
            beta[token],
            normalize,
        )
    state = original.clone()
    args = [al, a, b, dt, q, k, v, state, indices, normalize]
    out = native("gdn").gdn_decode(*args)
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.001)
    torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-6)
    state.copy_(original)
    assert torch.equal(out, native("gdn", raw=True).gdn_decode(*args))
    torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-6)

    starts = torch.tensor([0, 1, tokens, tokens], dtype=torch.int32)
    expected_state = original.clone()
    for slot in range(3):
        for token in range(starts[slot], starts[slot + 1]):
            expected[0, token], expected_state[slot] = _gdn_step(
                q[0, token],
                k[0, token],
                v[0, token],
                expected_state[slot],
                g[token].exp(),
                beta[token],
                normalize,
            )
    state.copy_(original)
    args = [q, k, v, g.unsqueeze(0), beta.unsqueeze(0), state, starts, normalize]
    out = native("gdn").gdn_prefill(*args)
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.001)
    torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-6)
    state.copy_(original)
    assert torch.equal(out, native("gdn", raw=True).gdn_prefill(*args))
    assert torch.equal(state[2], original[2])


@pytest.mark.parametrize("sigmoid", [False, True])
def test_arm_gdn_gated_norm_tail(native, raw, sigmoid):
    torch.manual_seed(441)
    x = torch.randn(7, 19, dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(19, dtype=torch.bfloat16)
    act = gate.float().sigmoid()
    if not sigmoid:
        act = gate.float() * act
    expected = (
        x.float()
        * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        * weight.float()
        * act
    ).bfloat16()
    args = [x, weight, gate, 1e-6, sigmoid]
    out = native("gdn").gdn_rmsnorm_gated(*args)
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.001)
    assert torch.equal(out, native("gdn", raw=True).gdn_rmsnorm_gated(*args))


@pytest.mark.parametrize("gate", [None, "silu", "sigmoid"])
@pytest.mark.parametrize("persistent", [False, True])
def test_arm_gdn_packed_output_and_state(native, raw, monkeypatch, gate, persistent):
    monkeypatch.setenv("FLAGGEMS_GDN_NATIVE_PERSISTENT_DECODE", str(int(persistent)))
    torch.manual_seed(816)
    kh, vh, kd, vd = 2, 4, 32, 16
    dim = kh * kd * 2 + vh * vd
    x = torch.randn(1, dim, dtype=torch.bfloat16) * 0.1
    untouched = x.clone()
    cs0 = torch.randn(2, dim, 3, dtype=torch.bfloat16) * 0.1
    rs0 = torch.randn(2, vh, vd, kd) * 0.01
    cw = torch.randn(dim, 4, dtype=torch.bfloat16) * 0.1
    cb = torch.randn(dim, dtype=torch.bfloat16) * 0.1
    a = torch.randn(1, vh, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    al, dt = torch.zeros(vh), torch.zeros(vh, dtype=torch.bfloat16)
    index = torch.tensor([1], dtype=torch.int32)
    cs, rs = cs0.clone(), rs0.clone()
    ops = native("gdn")
    mixed = ops.gdn_conv1d_update(x, cs, cw, cb, index, True)
    q, k, v = mixed.split([kh * kd, kh * kd, vh * vd], -1)
    expected = ops.gdn_decode(
        al,
        a,
        b,
        dt,
        q.view(1, 1, kh, kd),
        k.view(1, 1, kh, kd),
        v.view(1, 1, vh, vd),
        rs,
        index,
        True,
    ).view(1, vh, vd)
    z = torch.randn_like(expected)
    weight = torch.randn(vd, dtype=torch.bfloat16)
    if gate is not None:
        expected = ops.gdn_rmsnorm_gated(expected, weight, z, 1e-6, gate == "sigmoid")
    expected_cs, expected_rs = cs.clone(), rs.clone()
    output = (
        torch.empty(1, vh, vd * 2, dtype=torch.bfloat16)[:, :, ::2]
        if gate is None and persistent
        else torch.empty_like(expected)
    )
    for is_raw in (False, True):
        cs.copy_(cs0)
        rs.copy_(rs0)
        args = [x, a, b, al, dt, cs, cw, cb, rs, index]
        if gate is None:
            native("gdn", raw=is_raw).gdn_packed_decode(*args, output, True)
        else:
            native("gdn", raw=is_raw).gdn_packed_decode_norm(
                *args, z, weight, 1e-6, gate == "sigmoid", output, True
            )
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.001)
        assert torch.equal(cs, expected_cs)
        torch.testing.assert_close(rs, expected_rs, rtol=2e-5, atol=2e-6)
        assert torch.equal(x, untouched)
        assert torch.equal(rs[0], rs0[0])


@pytest.mark.parametrize("rows,mr", [(0, 1), (1, 1), (7, 4), (16, 4)])
def test_arm_silu_a8_rounding_and_row_stride(native, raw, rows, mr):
    from flag_gems.runtime.backend._arm.fused import (
        silu_mul_a8_pack_native,
        silu_mul_a8_pack_raw,
    )

    if not os.environ.get("FLAGGEMS_ARM_SILU_LIBRARY"):
        pytest.skip("build the optional Arm SiLU packing library")
    torch.manual_seed(554)
    k = 96
    gu = torch.randn(rows, 2 * k + 17)[:, : 2 * k]
    packed = silu_mul_a8_pack_native(gu, mr)
    assert torch.equal(packed, silu_mul_a8_pack_raw(gu, mr))
    if not rows:
        assert packed.numel() == 0
        return
    activation = torch.nn.functional.silu(gu[:, :k]) * gu[:, k:]
    low = torch.minimum(activation.amin(-1), torch.zeros(rows))
    high = torch.maximum(activation.amax(-1), torch.zeros(rows))
    multiplier = torch.where(high == low, 1.0, 255.0 / (high - low))
    use_min = -128.0 + low * multiplier + 127.0 + high * multiplier > 0
    zp = (
        torch.where(use_min, -128.0 - low * multiplier, 127.0 - high * multiplier)
        .clamp(-128, 127)
        .round()
        .int()
    )
    code = (
        ((activation * multiplier[:, None]).round().int() + zp[:, None])
        .clamp(-128, 127)
        .to(torch.int8)
    )
    panels = packed.reshape(-1, mr * (k + 8))
    actual = (
        panels[:, : mr * k]
        .contiguous()
        .view(torch.int8)
        .reshape(-1, k // 8, mr, 8)
        .permute(0, 2, 1, 3)
        .reshape(-1, k)[:rows]
    )
    assert torch.equal(actual, code)
    offsets = (
        panels[:, mr * k : mr * (k + 4)].contiguous().view(torch.int32).flatten()[:rows]
    )
    assert torch.equal(offsets, -zp)
    scales = panels[:, mr * (k + 4) :].contiguous().view(torch.float32).flatten()[:rows]
    torch.testing.assert_close(scales, 1.0 / multiplier, rtol=1e-6, atol=0)
