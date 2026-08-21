import os
import platform

import pytest
import torch


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires the macOS Arm libtriton_jit runtime",
)


def test_w8_prefill_stealing_matches_regular_grid(monkeypatch):
    runtime_library = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if not runtime_library or not os.path.isfile(runtime_library):
        pytest.skip("FLAGGEMS_LIBTRITON_JIT_Q4_OP is not configured")

    from flag_gems.csrc.arm import configure_runtime
    from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_qsi8cxp

    torch.ops.load_library(os.path.realpath(configure_runtime()))
    torch.manual_seed(42)
    m, n, k = 32, 64, 128
    x = torch.randn((m, k), dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n)
    rhs = pack_rhs_qsi8cxp(weight, scale)

    monkeypatch.setenv("FLAGGEMS_W8_STEALING_PREFILL", "0")
    regular = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)
    monkeypatch.setenv("FLAGGEMS_W8_STEALING_PREFILL", "1")
    monkeypatch.setenv("FLAGGEMS_W8_PREFILL_STEAL_CHUNK", "2")
    stealing = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)

    torch.testing.assert_close(stealing, regular, rtol=0, atol=0)
