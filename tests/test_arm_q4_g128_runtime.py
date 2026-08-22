import os
import platform

import pytest
import torch


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires the macOS Arm libtriton_jit runtime",
)


def test_g128_prefill_stealing_matches_regular_grid(monkeypatch):
    runtime_library = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if not runtime_library or not os.path.isfile(runtime_library):
        pytest.skip("FLAGGEMS_LIBTRITON_JIT_Q4_OP is not configured")

    from flag_gems.csrc.arm import configure_runtime

    torch.ops.load_library(os.path.realpath(configure_runtime()))
    torch.manual_seed(42)
    m, n, k = 32, 64, 128
    x = torch.randn((m, k), dtype=torch.bfloat16)
    groups = k // 128
    tile_stride = groups * 264 + 16
    rhs = torch.randint(
        0, 256, (n // 4, tile_stride), dtype=torch.uint8
    )
    scales = torch.ones(4, dtype=torch.bfloat16).view(torch.uint8)
    rhs[:, :8] = scales
    rhs[:, groups * 264 :] = 0
    rhs = rhs.flatten()

    monkeypatch.setenv("FLAGGEMS_ARM_Q4_G128_PREFILL_BLOCK_M", "16")
    monkeypatch.setenv("FLAGGEMS_ARM_Q4_G128_PREFILL_SUBGROUP_UNROLL", "1")
    monkeypatch.setenv("FLAGGEMS_Q4_PREFILL_THREADS", "2")
    monkeypatch.setenv("FLAGGEMS_ARM_Q4_G128_STEAL_CHUNK", "2")
    monkeypatch.setenv("FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL", "0")
    regular = torch.ops.triton_jit_cpu.q4_linear_g128(x, rhs, n, k)
    monkeypatch.setenv("FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL", "1")
    stealing = torch.ops.triton_jit_cpu.q4_linear_g128(x, rhs, n, k)

    torch.testing.assert_close(stealing, regular, rtol=0, atol=0)


def test_g128_decode_is_repeatable():
    runtime_library = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if not runtime_library or not os.path.isfile(runtime_library):
        pytest.skip("FLAGGEMS_LIBTRITON_JIT_Q4_OP is not configured")

    from flag_gems.csrc.arm import configure_runtime

    torch.ops.load_library(os.path.realpath(configure_runtime()))
    torch.manual_seed(43)
    n, k = 64, 128
    x = torch.randn((1, k), dtype=torch.bfloat16)
    groups = k // 128
    tile_stride = groups * 264 + 16
    rhs = torch.randint(
        0, 256, (n // 4, tile_stride), dtype=torch.uint8
    )
    scales = torch.ones(4, dtype=torch.bfloat16).view(torch.uint8)
    rhs[:, :8] = scales
    rhs[:, groups * 264 :] = 0
    rhs = rhs.flatten()

    first = torch.ops.triton_jit_cpu.q4_linear_g128(x, rhs, n, k)
    second = torch.ops.triton_jit_cpu.q4_linear_g128(x, rhs, n, k)

    torch.testing.assert_close(second, first, rtol=0, atol=0)
