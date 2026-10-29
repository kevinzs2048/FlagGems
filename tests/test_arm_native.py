# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional Arm native/TLE operators, independent numerical and state oracles."""

import ctypes
import inspect
import os
import platform

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in {"arm64", "aarch64"},
    reason="requires AArch64",
)


@pytest.fixture(scope="module")
def native():
    from flag_gems.runtime.backend._arm.tle_raw.silu_mul_a8_pack_native import (
        _ensure_native_library,
    )

    directory = (
        Path(__file__).resolve().parents[1]
        / "src/flag_gems/runtime/backend/_arm/tle_raw/lib"
    )
    if not os.getenv("FLAGGEMS_ARM_SILU_LIBRARY") and not list(
        directory.glob("libflag_gems_arm_silu_native.*")
    ):
        if os.getenv("FLAGGEMS_ARM_REQUIRE_NATIVE_TESTS") == "1":
            pytest.fail("Strict Arm test job requires a built SiLU leaf library")
        pytest.skip("build the optional Arm SiLU leaf library")
    _ensure_native_library()


@pytest.fixture
def raw(native):
    if os.environ.get("FLAGGEMS_ARM_REQUIRE_NATIVE_TESTS") == "1":
        from triton.language.extra.cpu import tle_ops as ops
    else:
        ops = pytest.importorskip("triton.language.extra.cpu.tle_ops")
    if "result_dtype" not in inspect.signature(ops.raw_call).parameters:
        if os.environ.get("FLAGGEMS_ARM_REQUIRE_NATIVE_TESTS") == "1":
            pytest.fail("Strict Arm test job requires the scalar CPU Raw compiler")
        pytest.skip("requires CPU scalar native-call support with void results")
    return native


def test_arm_silu_raw_rejects_old_single_program_abi(native, raw, monkeypatch):
    from types import SimpleNamespace

    from flag_gems.runtime.backend._arm.tle_raw import silu_a8_raw

    silu_a8_raw.initialize_silu_a8_raw()
    _, compiled, _ = silu_a8_raw._owners[:3]
    stale = SimpleNamespace(
        name=compiled.name,
        metadata_group=compiled.metadata_group,
        metadata=SimpleNamespace(
            single_program_name=compiled.metadata.single_program_name
        ),
    )
    monkeypatch.setattr(silu_a8_raw, "_owners", None)
    monkeypatch.setattr(
        silu_a8_raw._silu_a8_raw_kernel, "warmup", lambda *a, **kw: stale
    )
    with pytest.raises(RuntimeError, match="scalar return"):
        silu_a8_raw.initialize_silu_a8_raw()
    assert silu_a8_raw._owners is None


def test_arm_silu_raw_uninitialized_entry_errors(native, raw):
    from flag_gems.runtime.backend._arm.tle_raw import silu_a8_raw

    silu_a8_raw.initialize_silu_a8_raw()
    library, _, _, variants = silu_a8_raw._owners
    reset = library.flag_gems_arm_register_silu_raw_kernel
    reset.argtypes = [ctypes.c_uint64]
    reset.restype = None
    register = library.flag_gems_arm_register_silu_raw_variant
    register.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
    register.restype = None
    reset(0)
    try:
        for rows, mr in ((1, 1), (1, 4), (4, 4)):
            with pytest.raises(RuntimeError, match="Initialize"):
                torch.ops.flag_gems_arm_silu_native.pack_raw.default(
                    torch.zeros(rows, 64), mr
                )
    finally:
        if len(variants) == 1:
            reset(variants[0][3])
        else:
            for mr, _, _, pointer in variants:
                register(mr, pointer)
    assert torch.equal(
        silu_a8_raw.silu_mul_a8_pack_raw(torch.zeros(1, 64), 4),
        torch.ops.flag_gems_arm_silu_native.pack(torch.zeros(1, 64), 4),
    )


def test_arm_silu_raw_rejects_native_abi_mismatch(native, raw, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from flag_gems.runtime.backend._arm.tle_raw import silu_a8_raw

    module = importlib.import_module(
        "flag_gems.runtime.backend._arm.tle_raw.silu_mul_a8_pack_native"
    )
    monkeypatch.setattr(silu_a8_raw, "_owners", None)
    monkeypatch.setattr(
        module,
        "_global_native_library",
        SimpleNamespace(flag_gems_arm_silu_raw_abi_version=lambda: 0),
    )
    with pytest.raises(RuntimeError, match="incompatible Raw ABI"):
        silu_a8_raw.initialize_silu_a8_raw()
    assert silu_a8_raw._owners is None


def test_arm_silu_raw_concurrent_initialization(native, raw, monkeypatch):
    import importlib
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from flag_gems.runtime.backend._arm.tle_raw import silu_a8_raw

    module = importlib.import_module(
        "flag_gems.runtime.backend._arm.tle_raw.silu_mul_a8_pack_native"
    )
    silu_a8_raw.initialize_silu_a8_raw()
    entered, release, second_started, second_done = (Event() for _ in range(4))
    real_register = module._global_native_library.flag_gems_arm_register_silu_raw_kernel

    real_variant = module._global_native_library.flag_gems_arm_register_silu_raw_variant

    def delayed_register(*args):
        entered.set()
        assert release.wait(10), "registration was not released"
        (real_register if len(args) == 1 else real_variant)(*args)

    real_version = module._global_native_library.flag_gems_arm_silu_raw_abi_version

    class DelayedLibrary:
        flag_gems_arm_silu_raw_abi_version = staticmethod(real_version)
        flag_gems_arm_register_silu_raw_kernel = staticmethod(delayed_register)
        flag_gems_arm_register_silu_raw_variant = staticmethod(delayed_register)

    monkeypatch.setattr(module, "_global_native_library", DelayedLibrary())
    monkeypatch.setattr(silu_a8_raw, "_owners", None)

    def second_initialize():
        second_started.set()
        silu_a8_raw.initialize_silu_a8_raw()
        second_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(silu_a8_raw.initialize_silu_a8_raw)
        try:
            assert entered.wait(10)
            second = pool.submit(second_initialize)
            assert second_started.wait(10)
            assert not second_done.wait(0.05), "readiness published before registration"
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    value = torch.randn(1, 64)
    assert torch.equal(
        silu_a8_raw.silu_mul_a8_pack_raw(value, 1),
        module.silu_mul_a8_pack_native(value, 1),
    )


@pytest.mark.parametrize(
    "rows,mr", [(0, 1), (1, 1), (1, 4), (2, 4), (3, 4), (7, 4), (16, 4)]
)
def test_arm_silu_a8_rounding_and_row_stride(native, raw, rows, mr):
    from flag_gems.runtime.backend._arm.fused import (
        silu_mul_a8_pack_native,
        silu_mul_a8_pack_raw,
    )

    if not os.environ.get("FLAGGEMS_ARM_SILU_LIBRARY"):
        pytest.skip("build the optional Arm SiLU packing library")
    torch.manual_seed(554)
    k = 96
    # The prepared entry is also used for views whose base is not 16-byte
    # aligned. The odd row stride exercises every FP32 alignment class.
    gu = torch.randn(rows, 2 * k + 17)[:, 3 : 2 * k + 3]
    if rows:
        # Exercise the external SLEEF fallback as well as the inline polynomial.
        gu[0, :8] = torch.tensor([-100.0, -81.0, -80.0, 0.0, 80.0, 81.0, 100.0, 0.5])
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

    padded_codes = (
        panels[:, : mr * k]
        .contiguous()
        .view(torch.int8)
        .reshape(-1, k // 8, mr, 8)
        .permute(0, 2, 1, 3)
        .reshape(-1, k)[rows:]
    )
    padded_offsets = (
        panels[:, mr * k : mr * (k + 4)].contiguous().view(torch.int32).flatten()[rows:]
    )
    padded_scales = (
        panels[:, mr * (k + 4) :].contiguous().view(torch.float32).flatten()[rows:]
    )
    assert torch.all(padded_codes == 127)
    assert torch.all(padded_offsets == -127)
    assert torch.all(padded_scales == 1)


def test_arm_silu_boundary_leaf_ownership(native, raw):
    # The composite library is loaded by raw first. A boundary-only bitcode
    # call must still resolve the standalone library's arithmetic body.
    from flag_gems.runtime.backend._arm.tle_raw import silu_a8_raw

    silu_a8_raw.initialize_silu_a8_raw()
    owner = silu_a8_raw._owners[0]
    leaf = owner.flag_gems_arm_silu_standalone_leaf
    address = ctypes.cast(leaf, ctypes.c_void_p).value
    global_leaf = ctypes.CDLL(None).flag_gems_arm_silu_standalone_leaf
    assert ctypes.cast(global_leaf, ctypes.c_void_p).value == address
    if sys.platform.startswith("linux"):
        original = owner.flag_gems_arm_silu_mul_a8_pack
        assert ctypes.cast(original, ctypes.c_void_p).value == address
    leaf.restype = ctypes.c_int32
    leaf.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int32] * 4
    torch.manual_seed(7319)
    for rows in (1, 3):
        x = torch.randn(rows, 137)[:, 1:129]
        for mr in (1, 4):
            expected = torch.ops.flag_gems_arm_silu_native.pack.default(x, mr)
            output = torch.empty_like(expected)
            assert leaf(x.data_ptr(), output.data_ptr(), rows, 137, 64, mr) == 0
            assert torch.equal(output, expected)
            assert torch.equal(silu_a8_raw.silu_mul_a8_pack_raw(x, mr), expected)
            output.fill_(165)
            assert leaf(x.data_ptr(), output.data_ptr(), rows, 137, 63, mr) == -1
            assert torch.all(output == 165)


def test_arm_silu_metadata_registration_stays_out_of_compute(native, raw, monkeypatch):
    from flag_gems.runtime.backend._arm.tle_raw import meta, silu_a8_raw
    from flag_gems.runtime.backend._arm.tle_raw.silu_mul_a8_pack_native import (
        silu_mul_a8_pack_native,
    )

    x = torch.randn(1, 128)
    expected = silu_mul_a8_pack_native(x)
    assert torch.equal(silu_a8_raw.silu_mul_a8_pack_raw(x), expected)

    def unexpected_registration():
        raise AssertionError("Fake registration must finish before compute")

    monkeypatch.setattr(meta, "register_arm_fake_kernels", unexpected_registration)
    assert torch.equal(silu_mul_a8_pack_native(x), expected)
    assert torch.equal(silu_a8_raw.silu_mul_a8_pack_raw(x), expected)
