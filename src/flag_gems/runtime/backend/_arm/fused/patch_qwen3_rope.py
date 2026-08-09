"""Qwen3 BF16 RoPE routes backed by ordinary Triton code generation.

Decode applies Q and K rotation in one launch.  Prefill uses one out-of-place
kernel that also materializes the contiguous attention layout.  Both preserve
the eager BF16 intermediate rounding and expose all arithmetic to LLVM; no
TLE_raw or external compute runtime is used.

The ordinary-Triton frequency-table generator is controlled by an explicit
flag and enabled by the Q4 model optimizer.  It is exact for the tested
standard Qwen3 configuration and faster than the eager matmul/cos/sin graph
on CIX.  Unsupported shapes and dtypes fall back to Transformers.
"""
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle

from .aot_rope_backend import create_aot_rope_backend
from ..profile_range import profile_range
from ..vector_config import ELEMENTWISE_TILE, NONLINEAR_ROLLED_TILE

logger = logging.getLogger(__name__)

_PATCHED: dict = {}
_ROTARY_PATCHED: dict = {}
_BLOCK_HALF = ELEMENTWISE_TILE
_AOT_BACKENDS: dict[tuple[int, int, int], object | None] = {}


@triton.jit
def _rope_frequency_bf16_kernel(
    position_ids_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    position_stride,
    attention_scaling,
    HEAD_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Generate Qwen's repeated-half RoPE cos/sin tables in one kernel."""
    token = tl.program_id(0)
    half: tl.constexpr = HEAD_DIM // 2
    position = tl.load(
        position_ids_ptr + token * position_stride
    ).to(tl.float32)
    lanes = tl.arange(0, BLOCK_HALF)
    for base in tl.range(
        0, half, BLOCK_HALF, loop_unroll_factor=1
    ):
        offsets = base + lanes
        valid = offsets < half
        inv_freq = tl.load(
            inv_freq_ptr + offsets, mask=valid, other=0.0
        ).to(tl.float32)
        angle = position * inv_freq
        cosine = (tl.cos(angle) * attention_scaling).to(tl.bfloat16)
        sine = (tl.sin(angle) * attention_scaling).to(tl.bfloat16)
        output = token * HEAD_DIM + offsets
        tl.store(cos_ptr + output, cosine, mask=valid)
        tl.store(cos_ptr + output + half, cosine, mask=valid)
        tl.store(sin_ptr + output, sine, mask=valid)
        tl.store(sin_ptr + output + half, sine, mask=valid)


@triton.jit
def _rope_qk_bf16_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    n_heads_q,
    head_dim,
    half,
    BLOCK_HALF: tl.constexpr,
):
    """Apply RoPE in-place to q (heads 0..n_heads_q) and k (heads n_heads_q..total)
    in a single kernel launch. Grid is (n_heads_q + n_heads_kv,).

    Layout: q_ptr / k_ptr point to flat memory [n_heads * head_dim] bf16 each.
    cos_ptr / sin_ptr point to [half] bf16 (interleaved RoPE convention:
    cos/sin first half repeated for the second half).
    """
    pid = tle.program_id(0)
    is_q = pid < n_heads_q
    # Branch on q vs k by selecting the right base + index
    row = tl.where(is_q, q_ptr + pid * head_dim, k_ptr + (pid - n_heads_q) * head_dim)
    for off in range(0, half, BLOCK_HALF):
        d = off + tl.arange(0, BLOCK_HALF)
        mask = d < half
        q0 = tl.load(row + d, mask=mask, other=0.0).to(tl.float32)
        q1 = tl.load(row + half + d, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + d, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + d, mask=mask, other=0.0).to(tl.float32)
        # Match eager PyTorch BF16 semantics: each multiply materializes a
        # BF16 result before the final BF16 add/sub.  Keeping all arithmetic
        # in FP32 changes roughly one third of values by one BF16 ULP and can
        # eventually change greedy decoding despite a small absolute error.
        q0c = (q0 * c).to(tl.bfloat16)
        q1s = (q1 * s).to(tl.bfloat16)
        q0s = (q0 * s).to(tl.bfloat16)
        q1c = (q1 * c).to(tl.bfloat16)
        r0 = q0c.to(tl.float32) - q1s.to(tl.float32)
        r1 = q0s.to(tl.float32) + q1c.to(tl.float32)
        tl.store(row + d, r0.to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(row + half + d, r1.to(q_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rope_qk_bf16_rows_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_ptr,
    sin_ptr,
    q_stride_h,
    q_stride_t,
    k_stride_h,
    k_stride_t,
    n_heads_q,
    n_tokens,
    head_dim,
    half,
    BLOCK_HALF: tl.constexpr,
):
    """Out-of-place fused Q/K RoPE for a B=1 prefill sequence."""
    pid = tle.program_id(0)
    head = pid // n_tokens
    token = pid - head * n_tokens
    is_q = head < n_heads_q
    kv_head = head - n_heads_q
    source = tl.where(
        is_q,
        q_ptr + head * q_stride_h + token * q_stride_t,
        k_ptr + kv_head * k_stride_h + token * k_stride_t,
    )
    output = tl.where(
        is_q,
        q_out_ptr + (head * n_tokens + token) * head_dim,
        k_out_ptr + (kv_head * n_tokens + token) * head_dim,
    )
    cos_row = cos_ptr + token * head_dim
    sin_row = sin_ptr + token * head_dim
    for off in range(0, half, BLOCK_HALF):
        d = off + tl.arange(0, BLOCK_HALF)
        mask = d < half
        q0 = tl.load(source + d, mask=mask, other=0.0).to(tl.float32)
        q1 = tl.load(source + half + d, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_row + d, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_row + d, mask=mask, other=0.0).to(tl.float32)
        q0c = (q0 * c).to(tl.bfloat16)
        q1s = (q1 * s).to(tl.bfloat16)
        q0s = (q0 * s).to(tl.bfloat16)
        q1c = (q1 * c).to(tl.bfloat16)
        r0 = q0c.to(tl.float32) - q1s.to(tl.float32)
        r1 = q0s.to(tl.float32) + q1c.to(tl.float32)
        tl.store(output + d, r0.to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(
            output + half + d,
            r1.to(q_ptr.dtype.element_ty),
            mask=mask,
        )


_PREWARM_DONE = False


def _prewarm():
    global _PREWARM_DONE
    if _PREWARM_DONE:
        return
    try:
        for hd in (128,):
            q = torch.zeros((2, hd), dtype=torch.bfloat16)
            k = torch.zeros((2, hd), dtype=torch.bfloat16)
            c = torch.zeros(hd // 2, dtype=torch.bfloat16)
            s = torch.zeros(hd // 2, dtype=torch.bfloat16)
            _rope_qk_bf16_kernel[(4,)](
                q,
                k,
                c,
                s,
                2,
                hd,
                hd // 2,
                BLOCK_HALF=_BLOCK_HALF,
                num_warps=1,
                num_stages=1,
            )
    except Exception:
        logger.debug("rope prewarm failed", exc_info=True)
    _PREWARM_DONE = True


def _rope_bf16_jit(q, k, cos_half, sin_half, n_heads_q, n_heads_kv, head_dim):
    """Apply RoPE in-place via single @triton.jit kernel launch (q+k fused)."""
    _prewarm()
    half = head_dim // 2
    total = n_heads_q + n_heads_kv
    aot_key = (n_heads_q, n_heads_kv, head_dim)
    if aot_key not in _AOT_BACKENDS:
        _AOT_BACKENDS[aot_key] = create_aot_rope_backend(*aot_key)
    aot = _AOT_BACKENDS[aot_key]
    if aot is not None:
        with profile_range("triton::rope_qk_ordinary_aot"):
            aot(q, k, cos_half, sin_half)
        return
    with profile_range("triton::rope_qk"):
        _rope_qk_bf16_kernel[(total,)](
            q,
            k,
            cos_half,
            sin_half,
            n_heads_q,
            head_dim,
            half,
            BLOCK_HALF=_BLOCK_HALF,
            num_warps=1,
            num_stages=1,
        )


def _patched_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Use ordinary Triton for B=1 BF16 decode and prefill Q/K RoPE."""
    # Fast path conditions
    if (
        q.dim() == 4
        and k.dim() == 4
        and q.shape[0] == 1
        and k.shape[0] == 1
        and q.shape[2] == k.shape[2]
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and cos.dim() in (2, 3, 4)
        and sin.dim() in (2, 3, 4)
    ):
        n_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        head_dim = q.shape[3]
        # Kernel uses interleaved convention: only reads cos/sin first half (assumes
        # cos[d/2:]==cos[:d/2] which is HF's standard repeat-half RoPE pattern).
        if cos.stride(-1) == 1 and sin.stride(-1) == 1:
            cos_half = cos.as_strided(
                (head_dim // 2,), (1,), cos.storage_offset()
            )
            sin_half = sin.as_strided(
                (head_dim // 2,), (1,), sin.storage_offset()
            )
        else:
            cos_half = (
                cos.reshape(-1, head_dim)[0, : head_dim // 2].contiguous()
            )
            sin_half = (
                sin.reshape(-1, head_dim)[0, : head_dim // 2].contiguous()
            )
        n_tokens = q.shape[2]
        if n_tokens == 1:
            # Q/K arrive as transpose views in Hugging Face attention.  The
            # decode kernel consumes a packed [heads, head_dim] layout while
            # preserving the original functional semantics.
            q_buf = q.contiguous()
            k_buf = k.contiguous()
            _rope_bf16_jit(
                q_buf,
                k_buf,
                cos_half,
                sin_half,
                n_heads,
                n_kv_heads,
                head_dim,
            )
            return q_buf, k_buf

        q_buf = torch.empty_like(q, memory_format=torch.contiguous_format)
        k_buf = torch.empty_like(k, memory_format=torch.contiguous_format)
        cos_rows = cos.reshape(-1, head_dim).contiguous()
        sin_rows = sin.reshape(-1, head_dim).contiguous()
        with profile_range("triton::rope_qk_prefill"):
            _rope_qk_bf16_rows_kernel[
                ((n_heads + n_kv_heads) * n_tokens,)
            ](
                q,
                k,
                q_buf,
                k_buf,
                cos_rows,
                sin_rows,
                q.stride(1),
                q.stride(2),
                k.stride(1),
                k.stride(2),
                n_heads,
                n_tokens,
                head_dim,
                head_dim // 2,
                BLOCK_HALF=_BLOCK_HALF,
                num_warps=1,
                num_stages=1,
            )
        return q_buf, k_buf

    # Fallback: original PyTorch implementation
    return _PATCHED["original"](q, k, cos, sin, unsqueeze_dim)


def _patched_rotary_embedding_forward(self, x, position_ids):
    """Generate the standard Qwen3 BF16 RoPE table with ordinary Triton."""
    if (
        x.device.type == "cpu"
        and x.dtype == torch.bfloat16
        and getattr(self, "rope_type", None) == "default"
        and position_ids.device.type == "cpu"
        and position_ids.dim() == 2
        and position_ids.shape[0] == 1
        and position_ids.numel() > 0
        and self.inv_freq.device.type == "cpu"
        and self.inv_freq.dtype == torch.float32
    ):
        head_dim = int(self.inv_freq.numel() * 2)
        positions = position_ids.contiguous()
        cos = torch.empty(
            (1, positions.shape[1], head_dim),
            dtype=torch.bfloat16,
            device=x.device,
        )
        sin = torch.empty_like(cos)
        with profile_range("triton::rope_frequency"):
            _rope_frequency_bf16_kernel[(positions.shape[1],)](
                positions,
                self.inv_freq,
                cos,
                sin,
                positions.stride(1),
                float(self.attention_scaling),
                HEAD_DIM=head_dim,
                BLOCK_HALF=NONLINEAR_ROLLED_TILE,
                num_warps=1,
                num_stages=1,
            )
        return cos, sin
    return _ROTARY_PATCHED["original"](self, x, position_ids)


def patch_qwen3_rope() -> int:
    """Monkey-patch apply_rotary_pos_emb in transformers.models.qwen3.

    Returns count of patched modules.
    """
    # Targets regular Qwen3 only. Qwen3.5 supports partial rotary
    # (q_rot vs q_pass split); needs separate handling.
    targets = [
        "transformers.models.qwen3.modeling_qwen3",
    ]
    n = 0
    for modname in targets:
        try:
            mod = __import__(modname, fromlist=["apply_rotary_pos_emb"])
        except (ImportError, AttributeError):
            continue
        if not hasattr(mod, "apply_rotary_pos_emb"):
            continue
        if modname in _PATCHED:
            continue
        original = getattr(mod, "apply_rotary_pos_emb")
        _PATCHED["original"] = original
        _PATCHED[modname] = True
        setattr(mod, "apply_rotary_pos_emb", _patched_apply_rotary_pos_emb)
        n += 1
        logger.info(f"Patched {modname}.apply_rotary_pos_emb")

        rotary_cls = getattr(mod, "Qwen3RotaryEmbedding", None)
        frequency_codegen = os.getenv(
            "FLAGGEMS_ARM_ROPE_FREQUENCY_CODEGEN", "0"
        ).lower() in {"1", "true", "on"}
        if (
            frequency_codegen
            and rotary_cls is not None
            and modname not in _ROTARY_PATCHED
        ):
            _ROTARY_PATCHED["original"] = rotary_cls.forward
            _ROTARY_PATCHED[modname] = rotary_cls
            rotary_cls.forward = _patched_rotary_embedding_forward
            n += 1
            logger.info(f"Patched {modname}.Qwen3RotaryEmbedding.forward")
    return n


def unpatch_qwen3_rope() -> int:
    n = 0
    for modname in list(_PATCHED.keys()):
        if modname == "original":
            continue
        try:
            mod = __import__(modname, fromlist=["apply_rotary_pos_emb"])
        except (ImportError, AttributeError):
            continue
        if "original" in _PATCHED:
            setattr(mod, "apply_rotary_pos_emb", _PATCHED["original"])
        del _PATCHED[modname]
        n += 1
    for modname, rotary_cls in list(_ROTARY_PATCHED.items()):
        if modname == "original":
            continue
        if "original" in _ROTARY_PATCHED:
            rotary_cls.forward = _ROTARY_PATCHED["original"]
        del _ROTARY_PATCHED[modname]
        n += 1
    _ROTARY_PATCHED.pop("original", None)
    return n
