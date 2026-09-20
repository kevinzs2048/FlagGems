"""Experimental non-affine LayerNorm + BF16 modulation multiplication."""

from functools import lru_cache

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("layernorm_scale")


def enable_normscale(module):
    if not getattr(module, "_shared_modulation_enabled", False):
        raise ValueError(
            "LayerNorm/scale fusion requires the shared modulation adapter"
        )
    for block in module.transformer_blocks:
        for norm in [block.img_norm1, block.img_norm2]:
            if (
                not isinstance(norm, torch.nn.LayerNorm)
                or norm.elementwise_affine
                or len(norm.normalized_shape) != 1
            ):
                raise ValueError("Expected non-affine one-dimensional LayerNorm")
    native = ops()
    count = 0
    for block in module.transformer_blocks:
        if getattr(block, "_fused_normscale_enabled", False):
            continue
        block._fused_normscale_enabled = True
        block._fused_normscale_op = native.run
        count += 1
    return count
