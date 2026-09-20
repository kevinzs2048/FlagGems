"""Experimental KAI microkernel/order comparison, no model dispatch changes."""

from functools import lru_cache


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("bf16_variants")
