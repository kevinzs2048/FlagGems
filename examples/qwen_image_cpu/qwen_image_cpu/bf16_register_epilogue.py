"""Experimental BF16 output written directly by the SME kernel; default off."""

from functools import lru_cache


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("bf16_register_epilogue")


def enable(module):
    ops()
    count = 0
    for layer in module.modules():
        if getattr(layer, "_bf16_sme_enabled", False) and layer.bias is None:
            layer._bf16_register_epilogue_enabled = True
            count += 1
    return count


def select(module, enabled):
    for layer in module.modules():
        if hasattr(layer, "_bf16_register_epilogue_enabled"):
            layer._bf16_register_epilogue_enabled = bool(enabled)
