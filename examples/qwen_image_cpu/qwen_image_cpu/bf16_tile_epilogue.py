"""Experimental per-tile BF16 output; not enabled in model dispatch."""

from functools import lru_cache


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("bf16_tile_epilogue")
