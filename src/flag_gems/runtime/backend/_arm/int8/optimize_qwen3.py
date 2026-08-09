"""Curated Qwen3 W8A8 optimization entry point for ARM Triton CPU.

This intentionally does not call ``flag_gems.enable()``.  The full operator
table contains generic kernels that are not all profitable on CPU.  Instead,
it enables only the model patches and operator overrides measured as
net-positive for Qwen3 W8A8 inference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from .quantize_live import quantize_and_replace_linears
from .replace import replace_linears_with_tle_int8


def optimize_qwen3_w8a8(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    *,
    skip: Iterable[str] | None = None,
    enable_attention: bool = False,
    enable_argmax: bool = True,
    enable_qkv_fusion: bool = True,
    enable_qk_norm_fusion: bool = True,
) -> dict[str, object]:
    """Enable the measured Qwen3 W8A8 ARM CPU path in-place.

    Args:
        model: Hugging Face Qwen3 model to optimize.
        state_dict: Optional pre-quantized W8 state with ``weight`` and
            ``weight_scale`` entries.  When omitted, Linear weights are
            quantized per output channel in memory.
        skip: Linear module names to leave unchanged.
        enable_attention: Replace PyTorch SDPA with the ARM prefill and decode
            router.  Long-context decode uses staged BFDOT ordinary-Triton
            codegen.  This is off by default because changing the reduction
            order is numerically close but not bit-exact to ATen.
        enable_argmax: Replace the vocabulary argmax with the Triton CPU op.
        enable_qkv_fusion: Fuse decode Q/K/V W8 projections into one
            compiler-generated SDOT launch. Enabled by default after
            bit-exact operator validation and model-level profiling.
        enable_qk_norm_fusion: Run decode Q and K RMSNorm in one
            compiler-generated launch. Enabled by default; unsupported
            shapes preserve the two original calls.

    Returns:
        Setup counters and the names of process-global operator overrides.

    The model is mutated: eligible Linear modules become ``TLEInt8Linear`` and
    Qwen3 decode methods are patched to use fused ordinary-Triton kernels.
    """
    if state_dict is None:
        linears = quantize_and_replace_linears(model, skip=skip)
        quantization = "live_per_channel_w8"
    else:
        linears = replace_linears_with_tle_int8(model, state_dict, skip=skip)
        quantization = "prequantized_w8"

    # Patch the module classes only after Linear replacement: the MLP wrapper
    # discovers packed W8 projections and keeps a reference to them.
    from ..fused.patch_qwen3_layer_norm import patch_qwen3_layer_norm
    from ..fused.patch_qwen3_mlp import patch_qwen3_mlp
    from ..fused.patch_qwen3_rmsnorm import patch_qwen3_rmsnorm
    from ..fused.patch_qwen3_rope import patch_qwen3_rope

    patches = {
        "rope": patch_qwen3_rope(),
        "rmsnorm": patch_qwen3_rmsnorm(),
        "layer_norm": patch_qwen3_layer_norm(),
        "mlp_modules": patch_qwen3_mlp(model),
    }
    if enable_qkv_fusion:
        from ..fused.patch_qwen3_qkv import patch_qwen3_qkv

        patches["qkv_modules"] = patch_qwen3_qkv(model)
    if enable_qk_norm_fusion:
        # Install after the class-wide RMSNorm patch so each coordinator's
        # saved fallback remains the audited ordinary-Triton implementation.
        from ..fused.patch_qwen3_qk_norm import patch_qwen3_qk_norm

        patches["qk_norm_modules"] = patch_qwen3_qk_norm(model)

    overrides = []
    if enable_attention:
        overrides.append("scaled_dot_product_attention")
    if enable_argmax:
        # Quantized decode logits are model-owned finite values.  This selects
        # the lean one-vector argmax state while the generic aten override
        # keeps its NaN-correct path outside this curated model entry point.
        from ..ops.argmax import set_argmax_vocab_assume_finite

        set_argmax_vocab_assume_finite(True)
        overrides.append("argmax")
    if overrides:
        from ..ops import apply_arm_overrides

        apply_arm_overrides(include=overrides)

    return {
        "quantization": quantization,
        "int8_linears": linears,
        "qwen3_patches": patches,
        "arm_overrides": overrides,
    }
