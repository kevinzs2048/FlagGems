import copy
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


def test_w8_decode_output_is_aligned_and_repeatable():
    runtime_library = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if not runtime_library or not os.path.isfile(runtime_library):
        pytest.skip("FLAGGEMS_LIBTRITON_JIT_Q4_OP is not configured")

    from flag_gems.csrc.arm import configure_runtime
    from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_qsi8cxp

    torch.ops.load_library(os.path.realpath(configure_runtime()))
    torch.manual_seed(43)
    n, k = 64, 128
    x = torch.randn((1, k), dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n)
    rhs = pack_rhs_qsi8cxp(weight, scale)

    first = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)
    second = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)

    assert first.data_ptr() % 64 == 0
    assert second.data_ptr() % 64 == 0
    assert first.storage_offset() == 0
    assert second.storage_offset() == 0
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_w8_vllm_fast_apply_is_data_only_and_exact():
    runtime_library = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if not runtime_library or not os.path.isfile(runtime_library):
        pytest.skip("FLAGGEMS_LIBTRITON_JIT_Q4_OP is not configured")
    pytest.importorskip("vllm")

    from flag_gems.csrc.arm import configure_runtime
    from flag_gems.runtime.backend._arm.q4 import linear
    from vllm.model_executor.kernels.linear.scaled_mm import (
        Int8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
        CPUInt8ScaledMMLinearKernel,
    )

    torch.ops.load_library(os.path.realpath(configure_runtime()))
    linear._enable_vllm_dynamic_int8()
    previous = linear.set_vllm_fast_apply_enabled(True)
    try:
        config = Int8ScaledMMLinearLayerConfig(False, True, False)
        names = (
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        )
        kernel = CPUInt8ScaledMMLinearKernel(config, names)
        layer = torch.nn.Module()
        torch.manual_seed(44)
        n, k = 64, 128
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.randint(-127, 128, (n, k), dtype=torch.int8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                0.001 + 0.02 * torch.rand(n), requires_grad=False
            ),
        )
        layer.register_parameter("input_scale", None)
        layer.register_parameter("input_zero_point", None)
        layer.register_parameter("azp_adj", None)

        kernel.process_weights_after_loading(layer)
        prepared_rhs = kernel._flag_gems_w8_prepared_rhs
        assert prepared_rhs is layer.weight
        assert (
            prepared_rhs.untyped_storage().data_ptr()
            == layer.weight.untyped_storage().data_ptr()
        )
        copied, copied_layer = copy.deepcopy((kernel, layer))
        torch.testing.assert_close(
            copied._flag_gems_w8_prepared_rhs,
            prepared_rhs,
            rtol=0,
            atol=0,
        )
        assert copied._flag_gems_w8_prepared_rhs is copied_layer.weight
        assert (
            copied._flag_gems_w8_prepared_rhs.untyped_storage().data_ptr()
            == copied_layer.weight.untyped_storage().data_ptr()
        )

        x = torch.randn((1, k), dtype=torch.bfloat16)
        output = kernel.apply_weights(layer, x)
        reference = torch.ops.triton_jit_cpu.w8_linear_kai(
            x, layer.weight, n, k
        )
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    finally:
        linear.set_vllm_fast_apply_enabled(previous)
