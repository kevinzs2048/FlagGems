import copy
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


def test_prefill_thread_override_is_scoped_across_q4_routes(monkeypatch):
    runtime_library = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if not runtime_library or not os.path.isfile(runtime_library):
        pytest.skip("FLAGGEMS_LIBTRITON_JIT_Q4_OP is not configured")

    from flag_gems.csrc.arm import configure_runtime
    from flag_gems.runtime.backend._arm.q4.linear import (
        pack_rhs_qsi4c32p_asym_compact,
    )

    torch.ops.load_library(os.path.realpath(configure_runtime()))
    torch.manual_seed(44)
    m, n, k = 4, 1024, 128
    quantized = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n, k // 32)
    rhs = pack_rhs_qsi4c32p_asym_compact(quantized, scale)
    x = torch.randn((m, k), dtype=torch.bfloat16)
    joined = torch.randn((m, 2 * k), dtype=torch.bfloat16)

    monkeypatch.setenv("FLAGGEMS_Q4_PREFILL_THREADS", "2")
    original_threads = torch.get_num_threads()
    torch.ops.triton_jit_cpu.q4_linear_g32_asym_compact(x, rhs, n, k)
    assert torch.get_num_threads() == original_threads
    torch.ops.triton_jit_cpu.q4_linear_g32_asym_compact_swiglu(
        joined, rhs, n, k
    )
    assert torch.get_num_threads() == original_threads


def test_g128_vllm_fast_apply_preserves_parameter_identity_on_deepcopy():
    runtime_library = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if not runtime_library or not os.path.isfile(runtime_library):
        pytest.skip("FLAGGEMS_LIBTRITON_JIT_Q4_OP is not configured")
    pytest.importorskip("vllm")

    from flag_gems.csrc.arm import configure_runtime
    from flag_gems.runtime.backend._arm.q4 import linear
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
        Dynamic4bitLinearKernel,
    )
    from vllm.scalar_type import scalar_types

    torch.ops.load_library(os.path.realpath(configure_runtime()))
    linear._enable_vllm_dynamic4bit_g128()
    previous = linear.set_vllm_fast_apply_enabled(True)
    try:
        torch.manual_seed(45)
        n, k = 64, 128
        config = MPLinearLayerConfig(
            full_weight_shape=(k, n),
            partition_weight_shape=(k, n),
            weight_type=scalar_types.int4,
            act_type=torch.bfloat16,
            group_size=128,
            zero_points=False,
            has_g_idx=False,
        )
        kernel = Dynamic4bitLinearKernel(config, "weight", "weight_scale")
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.randint(-8, 8, (n, k), dtype=torch.int8),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                0.001 + 0.02 * torch.rand(n, k // 128),
                requires_grad=False,
            ),
        )

        kernel.process_weights_after_loading(layer)
        assert kernel._flag_gems_q4_prepared_rhs is layer.weight
        copied, copied_layer = copy.deepcopy((kernel, layer))
        assert copied._flag_gems_q4_prepared_rhs is copied_layer.weight

        x = torch.randn((1, k), dtype=torch.bfloat16)
        output = copied.apply_weights(copied_layer, x)
        reference = torch.ops.triton_jit_cpu.q4_linear_g128(
            x, copied_layer.weight, n, k
        )
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    finally:
        linear.set_vllm_fast_apply_enabled(previous)
