import copy
import platform

import pytest
import torch


pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in {"arm64", "aarch64"},
    reason="requires an AArch64 libtriton_jit runtime",
)


def _load_arm_runtime() -> None:
    from flag_gems.csrc.arm import configure_runtime

    try:
        runtime_library = configure_runtime()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    torch.ops.load_library(str(runtime_library.resolve()))


def _symmetric_w8_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    values = x.to(torch.float32)
    absmax = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    input_scale = absmax / 127.0
    quantized = (
        torch.round(values * (127.0 / absmax))
        .clamp_(-127, 127)
        .to(torch.float32)
    )
    accumulator = quantized @ weight.to(torch.float32).T
    return (
        accumulator * (input_scale * weight_scale.to(torch.float32)[None, :])
    ).to(torch.bfloat16)


def test_w8_prefill_stealing_matches_regular_grid(monkeypatch):
    from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_w8_symmetric

    _load_arm_runtime()
    torch.manual_seed(42)
    m, n, k = 32, 64, 128
    x = torch.randn((m, k), dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n)
    rhs = pack_rhs_w8_symmetric(weight, scale)

    monkeypatch.setenv("FLAGGEMS_W8_STEALING_PREFILL", "0")
    regular = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)
    monkeypatch.setenv("FLAGGEMS_W8_STEALING_PREFILL", "1")
    monkeypatch.setenv("FLAGGEMS_W8_PREFILL_STEAL_CHUNK", "2")
    stealing = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)

    reference = _symmetric_w8_reference(x, weight, scale)
    # RNE at exact half-way values may differ by one INT8 LSB between the
    # generated Arm conversion and torch.round.  The scale and symmetric
    # zero-point contract remain identical.
    torch.testing.assert_close(regular, reference, rtol=0.02, atol=0.125)
    torch.testing.assert_close(stealing, regular, rtol=0, atol=0)


@pytest.mark.parametrize("m", [3, 9, 13, 17, 25, 33])
def test_w8_compact_prefill_tails_match_reference(monkeypatch, m):
    from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_w8_symmetric

    _load_arm_runtime()
    torch.manual_seed(4200 + m)
    n, k = 64, 128
    x = torch.randn((m, k), dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n)
    rhs = pack_rhs_w8_symmetric(weight, scale)

    monkeypatch.setenv("FLAGGEMS_W8_STEALING_PREFILL", "1")
    monkeypatch.setenv("FLAGGEMS_W8_PREFILL_STEAL_CHUNK", "2")
    output = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)

    reference = _symmetric_w8_reference(x, weight, scale)
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.125)


def test_w8_prefill_thread_override_is_scoped(monkeypatch):
    from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_w8_symmetric

    _load_arm_runtime()
    torch.manual_seed(46)
    m, n, k = 32, 64, 128
    x = torch.randn((m, k), dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n)
    rhs = pack_rhs_w8_symmetric(weight, scale)

    monkeypatch.setenv("FLAGGEMS_W8_PREFILL_THREADS", "2")
    original_threads = torch.get_num_threads()
    torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)
    assert torch.get_num_threads() == original_threads


def test_w8_decode_output_is_aligned_and_repeatable():
    from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_w8_symmetric

    _load_arm_runtime()
    torch.manual_seed(43)
    n, k = 64, 128
    x = torch.randn((1, k), dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n)
    rhs = pack_rhs_w8_symmetric(weight, scale)

    first = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)
    second = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)

    assert first.data_ptr() % 64 == 0
    assert second.data_ptr() % 64 == 0
    assert first.storage_offset() == 0
    assert second.storage_offset() == 0
    reference = _symmetric_w8_reference(x, weight, scale)
    torch.testing.assert_close(first, reference, rtol=0.02, atol=0.125)
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_w8_body_decode_stealing_matches_static_grid(monkeypatch):
    from flag_gems.runtime.backend._arm.q4.linear import pack_rhs_w8_symmetric

    _load_arm_runtime()
    torch.manual_seed(45)
    n, k = 1024, 2048
    x = torch.randn((1, k), dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    scale = 0.001 + 0.02 * torch.rand(n)
    rhs = pack_rhs_w8_symmetric(weight, scale)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        monkeypatch.setenv("FLAGGEMS_W8_DECODE_PARTITIONS", "2")
        monkeypatch.setenv("FLAGGEMS_W8_STEALING_DECODE", "0")
        regular = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)

        monkeypatch.setenv("FLAGGEMS_W8_STEALING_DECODE", "1")
        monkeypatch.setenv("FLAGGEMS_W8_STEALING_MIN_WORK", "0")
        monkeypatch.setenv("FLAGGEMS_W8_BODY_STEAL_CHUNK", "1")
        stealing = torch.ops.triton_jit_cpu.w8_linear_kai(x, rhs, n, k)
    finally:
        torch.set_num_threads(previous_threads)

    torch.testing.assert_close(stealing, regular, rtol=0, atol=0)


def test_w8_vllm_fast_apply_is_data_only_and_exact():
    pytest.importorskip("vllm")

    from flag_gems.runtime.backend._arm.q4 import linear
    from vllm.model_executor.kernels.linear.scaled_mm import (
        Int8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
        CPUInt8ScaledMMLinearKernel,
    )

    _load_arm_runtime()
    linear._enable_vllm_dynamic_int8()
    previous = linear.set_vllm_fast_apply_enabled(True)
    try:
        config = Int8ScaledMMLinearLayerConfig(False, True, True)
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
