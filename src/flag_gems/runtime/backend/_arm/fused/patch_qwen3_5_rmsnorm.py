"""Monkey-patch Qwen3_5RMSNorm.forward with ordinary Triton codegen.

Per Qwen3.5-2B decode token, RMSNorm is invoked at:
  - input_layernorm (per decoder layer)            : 24 calls
  - post_attention_layernorm (per decoder layer)   : 24 calls
  - q_norm, k_norm (full_attention layers)         : 6 layers × 2 = 12 calls
Total: ~60 RMSNorm calls per token → ~1.8 ms/token saved if each call drops
from 30 us to 5 us.

Qwen3.5's RMSNorm uses `out = (x / rms(x)) * (1 + weight)` (note the +1).
The compiler sees the complete reduction and epilogue; no coarse runtime
operation or hand-written C loop is called.

Decode (M=1, BF16) hits the fast path. Other shapes / dtypes fall back
to the original forward.
"""
import logging
import types

import torch
import triton
import triton.language as tl

from ..vector_config import REDUCTION_TILE

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit(do_not_specialize=["eps"])
def _rms_norm_qwen35_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    D: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row * D
    lanes = tl.arange(0, BLOCK_SIZE)
    sum_sq = tl.zeros((1,), tl.float32)
    for base in range(0, D, BLOCK_SIZE):
        idx = base + lanes
        mask = idx < D
        x = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0).to(
            tl.float32
        )
        sum_sq += tl.sum(x * x, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / D + eps)
    for base in range(0, D, BLOCK_SIZE):
        idx = base + lanes
        mask = idx < D
        x = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        # Qwen3.5 keeps normalization and (1 + weight) multiplication in
        # FP32, then converts only the final result to the input dtype.
        tl.store(
            out_ptr + row_base + idx,
            x * rrms * (1.0 + weight),
            mask=mask,
        )


def _patched_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Fast path: bf16 input, last-dim contiguous, single row.
    if x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[-1] == self._triton_D:
        shape = x.shape
        D = self._triton_D
        M = x.numel() // D
        x_2d = x.reshape(M, D)
        out = torch.empty_like(x_2d)
        _rms_norm_qwen35_kernel[(M,)](
            x_2d,
            self.weight,
            out,
            D=D,
            eps=float(self.eps),
            BLOCK_SIZE=REDUCTION_TILE,
            num_warps=1,
            num_stages=1,
        )
        return out.reshape(*shape)

    # Slow / fallback path: original forward
    return self._original_forward(x)


def _get_qwen3_5_rmsnorm_classes():
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNorm"),
        ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5RMSNorm"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextRMSNorm"),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_rmsnorm(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) not in _PATCHED:
            D = mod.weight.shape[0]
            mod._triton_D = D
            mod._original_forward = mod.forward
            mod.forward = types.MethodType(_patched_rmsnorm_forward, mod)
            _PATCHED.add(id(mod))
            n += 1
    if n > 0:
        logger.info("Patched %d Qwen3.5 RMSNorm modules with Triton codegen", n)
    return n


def unpatch_qwen3_5_rmsnorm(model) -> int:
    rms_classes = _get_qwen3_5_rmsnorm_classes()
    if not rms_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, rms_classes) and id(mod) in _PATCHED:
            if hasattr(mod, "_original_forward"):
                mod.forward = mod._original_forward
                del mod._original_forward
                del mod._triton_D
            _PATCHED.discard(id(mod))
            n += 1
    return n
