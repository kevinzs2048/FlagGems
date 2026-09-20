"""Exact BF16 SiLU table and fused rounded-BF16 SiLU-times-up inference."""

from functools import lru_cache

import torch
from torch.nn import functional as F


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("pointwise_fusion")


@lru_cache(None)
def silu_table():
    with torch.inference_mode():
        encodings = (
            torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        )
        return F.silu(encodings).contiguous()
