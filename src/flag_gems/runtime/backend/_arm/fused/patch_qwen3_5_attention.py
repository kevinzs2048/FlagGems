"""Route Qwen3.5 SDPA through the audited Arm attention implementation.

M=1 decode uses the staged ordinary-Triton BFDOT schedule from N=512 by
default.  Shorter contexts use the legacy runtime when installed, otherwise
the compiler-visible online schedule.  Prefill uses the existing tiled Triton
kernel, while unsupported shapes retain the original ATen implementation.
``FLAGGEMS_ARM_ATTN_DECODE_IMPL`` can force a path for controlled A/B testing.
"""

import logging

import torch.nn.functional as F

# Import before installing the monkey patch.  attention.py captures the true
# ATen function at module initialization, so its fallback cannot recurse.
from ..ops.attention import (
    scaled_dot_product_attention as _arm_scaled_dot_product_attention,
)

logger = logging.getLogger(__name__)

_orig_sdpa = F.scaled_dot_product_attention
_PATCHED = False


def patch_qwen3_5_attention(model=None) -> int:
    """Install the Arm SDPA router; ``model`` is accepted for API symmetry."""
    global _PATCHED
    if _PATCHED:
        return 1
    F.scaled_dot_product_attention = _arm_scaled_dot_product_attention
    _PATCHED = True
    logger.info(
        "Patched F.scaled_dot_product_attention with Arm Triton codegen"
    )
    return 1


def unpatch_qwen3_5_attention(model=None) -> int:
    global _PATCHED
    if not _PATCHED:
        return 0
    F.scaled_dot_product_attention = _orig_sdpa
    _PATCHED = False
    return 1
