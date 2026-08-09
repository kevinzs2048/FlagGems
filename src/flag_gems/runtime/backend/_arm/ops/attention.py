"""ARM CPU attention schedules implemented with ordinary Triton operations.

Prefill uses tiled Flash Attention v2.  Decode uses either a single-kernel
online softmax or, for sufficiently long KV sequences, a two-kernel schedule:
QK/softmax is lowered to BFDOT and a register-bounded PV kernel consumes its
FP32 scratch.  The decode compute remains visible to Triton/LLVM; an optional
legacy C runtime is kept only for short-context A/B and compatibility.

The optimized paths support BF16 and GQA.  Unsupported shapes, masks and
dtypes fall back to ATen.
"""
import ctypes
import logging
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ..profile_range import profile_range

log = logging.getLogger(__name__)

# Preload libsleef (tl.math.exp2 in Triton-CPU .so depends on SLEEF symbols).


def _ensure_sleef():
    try:
        import triton as _t

        sleef_dir = os.path.join(os.path.dirname(_t.__file__), "_C")
        sleef_so = os.path.join(sleef_dir, "libsleef.so.3")
        if not os.path.exists(sleef_so):
            return
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if sleef_dir not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{sleef_dir}:{ld}"
        ctypes.CDLL(sleef_so)  # preload so later dlopen can resolve symbols
    except Exception:
        pass


_ensure_sleef()

# Keep the original ATen SDPA for internal fallback (avoids infinite recursion after monkey-patch).
_aten_sdpa = F.scaled_dot_product_attention

# Import once at module load. If triton-cpu lacks the runtime module (older
# build), fall through to ATen for M=1 decode.
try:
    from triton.language.extra.cpu.runtime import (
        flash_attn_decode_bf16 as _flash_attn_decode_bf16,
    )
except ImportError:
    _flash_attn_decode_bf16 = None

# log2(e) = 1/ln(2) — used so we can substitute exp2 for exp (avoids SLEEF precision loss).
_LOG2E: float = 1.44269504089

# Block sizes (BLOCK_N=16 chosen via sweep).
_BLOCK_M: int = 32
_BLOCK_N: int = 16


def _power_of_two_env(name: str, default: int, maximum: int) -> int:
    """Read a positive power-of-two launch parameter or fail explicitly."""
    value = int(os.getenv(name, str(default)))
    if value <= 0 or value > maximum or value & (value - 1):
        raise ValueError(
            f"{name} must be a power of two in [1, {maximum}], got {value}"
        )
    return value

# ── Flash Attention Triton Kernel ───────────────────────────────────────────


@triton.jit
def _flash_attn_decode_codegen_kernel(
    Q,
    K,
    V,
    Out,
    seqlen_k,
    sm_scale,
    q_numhead,
    kv_numhead,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """M=1 GQA attention expressed entirely in ordinary Triton operations."""
    q_head = tl.program_id(0)
    kv_head = q_head * kv_numhead // q_numhead
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_head * HEAD_DIM + offs_d).to(tl.float32)

    max_score = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    for start_n in range(0, seqlen_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid = offs_n < seqlen_k
        kv_offsets = (
            kv_head * seqlen_k * HEAD_DIM
            + offs_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        key = tl.load(
            K + kv_offsets, mask=valid[:, None], other=0.0
        ).to(tl.float32)
        score = tl.sum(key * q[None, :], axis=1) * sm_scale
        score = tl.where(valid, score, -float("inf"))

        block_max = tl.max(score, axis=0)
        next_max = tl.maximum(max_score, block_max)
        old_scale = tl.exp(max_score - next_max)
        probability = tl.exp(score - next_max)
        denominator = denominator * old_scale + tl.sum(
            probability, axis=0
        )

        value = tl.load(
            V + kv_offsets, mask=valid[:, None], other=0.0
        ).to(tl.float32)
        accumulator = (
            accumulator * old_scale
            + tl.sum(probability[:, None] * value, axis=0)
        )
        max_score = next_max

    tl.store(
        Out + q_head * HEAD_DIM + offs_d,
        (accumulator / denominator).to(tl.bfloat16),
    )


@triton.jit
def _flash_attn_decode_scores_codegen_kernel(
    Q,
    K,
    Scores,
    seqlen_k,
    sm_scale,
    q_numhead,
    kv_numhead,
    HEAD_DIM: tl.constexpr,
    SCORE_BLOCK: tl.constexpr,
):
    """Compute normalized-softmax numerators without a wide V accumulator.

    Keeping the QK reduction and the PV accumulation in separate kernels is
    intentional on CPU.  The online kernel otherwise keeps Q and a
    HEAD_DIM-wide FP32 accumulator live together and rescales the complete
    accumulator after every token.  Scores is caller-owned scratch and holds
    FP32 exp(score - max); Denominator contains one scalar per query head.
    """
    q_head = tl.program_id(0)
    kv_head = q_head * kv_numhead // q_numhead
    offs_d = tl.arange(0, HEAD_DIM)
    # Keep a leading row dimension so the ordinary BF16 mul+sum graph matches
    # the Arm backend's matrix-vector BFDOT pattern.  A rank-one reduction is
    # intentionally not used: it lowers to rounded BF16 multiplies instead of
    # the accumulating dot-product instruction.
    q = tl.load(Q + q_head * HEAD_DIM + offs_d)
    score_base = Scores + q_head * seqlen_k

    max_score = tl.full((), -float("inf"), tl.float32)
    for n in range(0, seqlen_k, 1):
        key = tl.load(
            K
            + kv_head * seqlen_k * HEAD_DIM
            + n * HEAD_DIM
            + offs_d
        )
        score_row = tl.sum(
            key[None, :] * q[None, :], axis=1, dtype=tl.float32
        ) * sm_scale
        tl.store(score_base + n + tl.arange(0, 1), score_row)
        max_score = tl.maximum(max_score, tl.max(score_row, axis=0))

    denominator = tl.zeros((), tl.float32)
    for start_n in range(0, seqlen_k, SCORE_BLOCK):
        offs_n = start_n + tl.arange(0, SCORE_BLOCK)
        valid = offs_n < seqlen_k
        score = tl.load(score_base + offs_n, mask=valid, other=-float("inf"))
        probability = tl.exp(score - max_score)
        tl.store(score_base + offs_n, probability, mask=valid)
        denominator += tl.sum(probability, axis=0)
    tl.store(Scores + q_numhead * seqlen_k + q_head, denominator)


@triton.jit
def _flash_attn_decode_pv_codegen_kernel(
    V,
    Scores,
    Out,
    seqlen_k,
    q_numhead,
    kv_numhead,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate one register-bounded output slice from staged scores."""
    pid = tl.program_id(0)
    blocks_d = HEAD_DIM // BLOCK_D
    q_head = pid // blocks_d
    block_d = pid % blocks_d
    kv_head = q_head * kv_numhead // q_numhead
    offs_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    score_base = Scores + q_head * seqlen_k
    value_base = V + kv_head * seqlen_k * HEAD_DIM

    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    for n in range(0, seqlen_k, 1):
        probability = tl.load(score_base + n)
        value = tl.load(value_base + n * HEAD_DIM + offs_d).to(tl.float32)
        accumulator += probability * value

    denominator = tl.load(Scores + q_numhead * seqlen_k + q_head)
    tl.store(
        Out + q_head * HEAD_DIM + offs_d,
        (accumulator / denominator).to(tl.bfloat16),
    )


@triton.jit
def _flash_attn_short_prefill_scores_codegen_kernel(
    Q,
    K,
    Scores,
    seqlen_q,
    seqlen_k,
    sm_scale,
    q_numhead,
    kv_numhead,
    HEAD_DIM: tl.constexpr,
    SCORE_BLOCK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Register-bounded QK/softmax for short B=1 prefill sequences."""
    pid = tl.program_id(0)
    q_head = pid // seqlen_q
    q_pos = pid - q_head * seqlen_q
    kv_head = q_head * kv_numhead // q_numhead
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        Q + (q_head * seqlen_q + q_pos) * HEAD_DIM + offs_d
    )
    score_base = Scores + pid * seqlen_k
    if IS_CAUSAL:
        valid_end = seqlen_k - seqlen_q + q_pos + 1
    else:
        valid_end = seqlen_k

    max_score = tl.full((), -float("inf"), tl.float32)
    for n in range(0, seqlen_k, 1):
        key = tl.load(
            K + (kv_head * seqlen_k + n) * HEAD_DIM + offs_d
        )
        score_row = tl.sum(
            key[None, :] * q[None, :], axis=1, dtype=tl.float32
        ) * sm_scale
        valid = n < valid_end
        score_row = tl.where(valid, score_row, -float("inf"))
        tl.store(score_base + n + tl.arange(0, 1), score_row)
        max_score = tl.maximum(max_score, tl.max(score_row, axis=0))

    denominator = tl.zeros((), tl.float32)
    for start_n in range(0, seqlen_k, SCORE_BLOCK):
        offs_n = start_n + tl.arange(0, SCORE_BLOCK)
        in_range = offs_n < seqlen_k
        valid = offs_n < valid_end
        score = tl.load(
            score_base + offs_n, mask=in_range, other=-float("inf")
        )
        probability = tl.where(valid, tl.exp(score - max_score), 0.0)
        # Materialize zeros for the causal suffix.  A scalar predicate on a
        # rolled-loop load is not reliably vectorized by all Triton-CPU/LLVM
        # combinations, while an unconditional initialized scratch row is.
        tl.store(score_base + offs_n, probability, mask=in_range)
        denominator += tl.sum(probability, axis=0)
    tl.store(Scores + q_numhead * seqlen_q * seqlen_k + pid, denominator)


@triton.jit
def _flash_attn_short_prefill_pv_codegen_kernel(
    V,
    Scores,
    Out,
    seqlen_q,
    seqlen_k,
    q_numhead,
    kv_numhead,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Consume short-prefill probabilities with a bounded D accumulator."""
    pid = tl.program_id(0)
    blocks_d = HEAD_DIM // BLOCK_D
    row = pid // blocks_d
    block_d = pid - row * blocks_d
    q_head = row // seqlen_q
    q_pos = row - q_head * seqlen_q
    kv_head = q_head * kv_numhead // q_numhead
    offs_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    score_base = Scores + row * seqlen_k
    value_base = V + kv_head * seqlen_k * HEAD_DIM
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    for n in range(0, seqlen_k, 1):
        probability = tl.load(score_base + n)
        value = tl.load(value_base + n * HEAD_DIM + offs_d).to(tl.float32)
        accumulator += probability * value

    denominator = tl.load(
        Scores + q_numhead * seqlen_q * seqlen_k + row
    )
    tl.store(
        Out + row * HEAD_DIM + offs_d,
        (accumulator / denominator).to(tl.bfloat16),
    )


@triton.jit
def _flash_attn_fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    # [B*Hq, M, D]
    stride_qh,
    stride_qm,
    stride_qk,
    # [B*Hkv, N, D]
    stride_kh,
    stride_kn,
    stride_kk,
    # [B*Hkv, N, D]
    stride_vh,
    stride_vn,
    stride_vk,
    # [B*Hq, M, D]
    stride_oh,
    stride_om,
    stride_ok,
    seqlen_q,
    seqlen_k,
    q_numhead,
    kv_numhead,  # GQA support
    LOG2E: tl.constexpr,  # 1.44269504
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,  # compile-time constant: generates two code paths
):
    pid_bh = tl.program_id(0)  # batch * Q-head (flattened)
    pid_m = tl.program_id(1)  # M-tile index

    # GQA mapping: every (Hq // Hkv) Q-heads share one KV-head.
    head_id = pid_bh % q_numhead
    batch_id = pid_bh // q_numhead
    kv_head_id = head_id * kv_numhead // q_numhead

    Q_bh = Q + (batch_id * q_numhead + head_id) * stride_qh
    K_bh = K + (batch_id * kv_numhead + kv_head_id) * stride_kh
    V_bh = V + (batch_id * kv_numhead + kv_head_id) * stride_vh
    O_bh = Out + (batch_id * q_numhead + head_id) * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    # Q: [BLOCK_M, HEAD_DIM], pre-multiplied by sm_scale*LOG2E (shifts into log2 domain).
    q = tl.load(
        Q_bh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32) * (sm_scale * LOG2E)

    # Online softmax state (per-row, log2 domain).
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    lse = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: only iterate up to the current Q-tile.
    if IS_CAUSAL:
        kv_end = tl.minimum(seqlen_k, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = seqlen_k

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen_k

        # K^T: [HEAD_DIM, BLOCK_N] — swapping k/n offsets gives the transposed load.
        k = tl.load(
            K_bh + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn,
            mask=mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # QK^T: [BLOCK_M, HEAD_DIM] x [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N].
        # q is already in log2 domain (includes sm_scale*LOG2E), so exp2 can be applied directly.
        qk = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16)).to(tl.float32)

        if IS_CAUSAL:
            causal_ok = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_ok & mask_n[None, :], qk, float("-inf"))
        else:
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

        # Online softmax (log2 domain).
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))  # [BLOCK_M]
        alpha = tl.math.exp2(m_i - m_new)  # rescale previous rows
        p = tl.math.exp2(qk - m_new[:, None])  # [BLOCK_M, BLOCK_N]

        lse = lse * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V: [BLOCK_N, HEAD_DIM]
        v = tl.load(
            V_bh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        # P @ V: [BLOCK_M, BLOCK_N] × [BLOCK_N, HEAD_DIM] → [BLOCK_M, HEAD_DIM]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        m_i = m_new

    # Normalize and write back.
    acc = acc / lse[:, None]
    tl.store(
        O_bh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None],
    )


# Python wrappers.


def _triton_flash_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Core Triton kernel invocation. Caller must have already verified the Triton path applies."""
    B, Hq, M, D = query.shape
    Hkv = key.shape[1]

    # Flatten batch+head -> [B*H, seq, D]
    q = query.reshape(B * Hq, M, D)
    k = key.reshape(B * Hkv, -1, D)
    v = value.reshape(B * Hkv, -1, D)
    N = k.shape[1]
    out = torch.empty_like(q)

    grid = (B * Hq, triton.cdiv(M, _BLOCK_M))

    _flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M,
        N,
        Hq,
        Hkv,
        _LOG2E,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=is_causal,
    )
    return out.reshape(B, Hq, M, D)


def _triton_flash_attn_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Compiler-visible M=1 attention; no TLE/raw or runtime call."""
    _, hq, _, head_dim = query.shape
    hkv = key.shape[1]
    seq_len = key.shape[2]
    # Decode tensors from Qwen have contiguous physical H/N/D storage even
    # when their size-one dimensions carry transpose-view strides.  Describe
    # that storage once instead of dispatching four squeeze operations and
    # two output unsqueezes per layer.
    if query.stride(1) == head_dim and query.stride(3) == 1:
        q_flat = query.as_strided(
            (hq, head_dim),
            (head_dim, 1),
            query.storage_offset(),
        )
    else:
        q_flat = query.squeeze(0).squeeze(1).contiguous()
    expected_kv_stride = (seq_len * head_dim, head_dim, 1)
    if key.stride()[1:] == expected_kv_stride:
        k_flat = key.as_strided(
            (hkv, seq_len, head_dim),
            expected_kv_stride,
            key.storage_offset(),
        )
    else:
        k_flat = key.squeeze(0).contiguous()
    if value.stride()[1:] == expected_kv_stride:
        v_flat = value.as_strided(
            (hkv, seq_len, head_dim),
            expected_kv_stride,
            value.storage_offset(),
        )
    else:
        v_flat = value.squeeze(0).contiguous()
    output = torch.empty((1, hq, 1, head_dim), dtype=torch.bfloat16)
    block_n = _power_of_two_env(
        "FLAGGEMS_ARM_ATTN_CODEGEN_BLOCK_N", 1, 64
    )
    with profile_range("triton::flash_attn_decode_codegen"):
        _flash_attn_decode_codegen_kernel[(hq,)](
            q_flat,
            k_flat,
            v_flat,
            output,
            seq_len,
            sm_scale,
            hq,
            hkv,
            HEAD_DIM=head_dim,
            BLOCK_N=block_n,
        )
    return output


def _triton_flash_attn_decode_staged(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Two-stage ordinary-Triton decode schedule for controlled CPU A/B."""
    _, hq, _, head_dim = query.shape
    hkv = key.shape[1]
    seq_len = key.shape[2]
    q_flat = query.squeeze(0).squeeze(1).contiguous()
    k_flat = key.squeeze(0).contiguous()
    v_flat = value.squeeze(0).contiguous()
    # Scores and the per-head denominator share one allocation.  Keeping the
    # denominator in a tail region avoids a second dispatcher/allocation cost
    # without hiding any compute behind a runtime helper.
    scores = torch.empty(hq * seq_len + hq, dtype=torch.float32)
    out_flat = torch.empty(hq, head_dim, dtype=torch.bfloat16)
    # Four FP32 lanes avoid saving a wide softmax vector across the SLEEF exp
    # call for longer short-prefill rows.  At T<=16, four independent exp
    # vectors amortize the call boundary slightly better on CIX.
    default_score_block = 16 if seq_len <= 16 else 4
    score_block = _power_of_two_env(
        "FLAGGEMS_ARM_ATTN_STAGED_SCORE_BLOCK",
        default_score_block,
        64,
    )
    block_d = _power_of_two_env(
        "FLAGGEMS_ARM_ATTN_STAGED_BLOCK_D", 64, head_dim
    )
    if head_dim % block_d:
        raise ValueError(
            "FLAGGEMS_ARM_ATTN_STAGED_BLOCK_D must divide HEAD_DIM "
            f"({head_dim}), got {block_d}"
        )
    with profile_range("triton::flash_attn_decode_staged"):
        _flash_attn_decode_scores_codegen_kernel[(hq,)](
            q_flat,
            k_flat,
            scores,
            seq_len,
            sm_scale,
            hq,
            hkv,
            HEAD_DIM=head_dim,
            SCORE_BLOCK=score_block,
        )
        _flash_attn_decode_pv_codegen_kernel[
            (hq * (head_dim // block_d),)
        ](
            v_flat,
            scores,
            out_flat,
            seq_len,
            hq,
            hkv,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
    return out_flat.unsqueeze(0).unsqueeze(2)


def _triton_flash_attn_short_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Two-stage ordinary-Triton attention for B=1, 1<M<BLOCK_M."""
    _, hq, seqlen_q, head_dim = query.shape
    hkv = key.shape[1]
    seqlen_k = key.shape[2]
    q_flat = query.contiguous().reshape(hq, seqlen_q, head_dim)
    k_flat = key.contiguous().reshape(hkv, seqlen_k, head_dim)
    v_flat = value.contiguous().reshape(hkv, seqlen_k, head_dim)
    rows = hq * seqlen_q
    scores = torch.empty(
        rows * seqlen_k + rows, dtype=torch.float32
    )
    out_flat = torch.empty(
        (hq, seqlen_q, head_dim), dtype=torch.bfloat16
    )
    score_block = _power_of_two_env(
        "FLAGGEMS_ARM_ATTN_STAGED_SCORE_BLOCK", 16, 64
    )
    block_d = _power_of_two_env(
        "FLAGGEMS_ARM_ATTN_STAGED_BLOCK_D", 64, head_dim
    )
    if head_dim % block_d:
        raise ValueError(
            "FLAGGEMS_ARM_ATTN_STAGED_BLOCK_D must divide HEAD_DIM "
            f"({head_dim}), got {block_d}"
        )
    with profile_range("triton::flash_attn_short_prefill"):
        _flash_attn_short_prefill_scores_codegen_kernel[(rows,)](
            q_flat,
            k_flat,
            scores,
            seqlen_q,
            seqlen_k,
            sm_scale,
            hq,
            hkv,
            HEAD_DIM=head_dim,
            SCORE_BLOCK=score_block,
            IS_CAUSAL=is_causal,
        )
        _flash_attn_short_prefill_pv_codegen_kernel[
            (rows * (head_dim // block_d),)
        ](
            v_flat,
            scores,
            out_flat,
            seqlen_q,
            seqlen_k,
            hq,
            hkv,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            IS_CAUSAL=is_causal,
        )
    return out_flat.unsqueeze(0)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """
    aten::scaled_dot_product_attention — ARM CPU Flash Attention.

    Triton prefill path conditions (otherwise use decode runtime or ATen):
      - dtype = bfloat16
      - attn_mask = None
      - dropout_p = 0.0
      - seqlen_q >= BLOCK_M (=32)
      - head_dim in {16,32,64,128,256}
    """
    B, Hq, M, D = query.shape

    short_prefill_codegen = os.getenv(
        "FLAGGEMS_ARM_ATTN_SHORT_PREFILL_CODEGEN", "0"
    ).lower() in {"1", "true", "on"}
    short_prefill_min = int(
        os.getenv("FLAGGEMS_ARM_ATTN_SHORT_PREFILL_MIN_SEQ", "10")
    )
    if (
        short_prefill_codegen
        and short_prefill_min <= M < _BLOCK_M
        and B == 1
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and D in {64, 128, 256}
    ):
        sm_scale = scale if scale is not None else D**-0.5
        return _triton_flash_attn_short_prefill(
            query, key, value, sm_scale, bool(is_causal)
        )

    # M=1 decode fast path.  Both compiler-generated schedules keep their
    # loops, reductions and vector math visible through LLVM.  Keep the legacy
    # C runtime selectable for controlled A/B measurements.
    # Requires BF16, no mask and no dropout.  Q/K/V produced by transformer
    # attention are commonly transpose views; the runtime path already packs
    # them into contiguous flat buffers below, so rejecting non-contiguous
    # inputs only disabled the fast path for real Hugging Face models.
    if (
        M == 1
        and B == 1
        and query.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and D in {64, 128, 256}
    ):
        Hkv = key.shape[1]
        seq_len = key.shape[2]
        sm_scale = scale if scale is not None else D**-0.5
        decode_impl = os.getenv(
            "FLAGGEMS_ARM_ATTN_DECODE_IMPL", "auto"
        ).lower()
        if decode_impl == "auto":
            # The staged ordinary-Triton schedule removes the wide online
            # accumulator and wins from N=512 with the production eight-core
            # setting.  Below that threshold a present legacy runtime still
            # wins on many cores; a stock Triton-CPU build stays entirely on
            # the compiler-visible online kernel.  The threshold remains
            # overrideable for single-thread deployments, where staged wins
            # even at N=128.
            staged_min_seq = int(
                os.getenv("FLAGGEMS_ARM_ATTN_STAGED_MIN_SEQ", "512")
            )
            disable_runtime = os.getenv(
                "FLAGGEMS_ARM_ATTN_DISABLE_RUNTIME", "0"
            ).lower() in {"1", "true", "on"}
            if seq_len >= staged_min_seq:
                decode_impl = "staged"
            elif disable_runtime or _flash_attn_decode_bf16 is None:
                decode_impl = "codegen"
            else:
                decode_impl = "runtime"
        if decode_impl == "codegen":
            return _triton_flash_attn_decode(
                query, key, value, sm_scale
            )
        if decode_impl == "staged":
            return _triton_flash_attn_decode_staged(
                query, key, value, sm_scale
            )
        if decode_impl == "aten":
            return _aten_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        if _flash_attn_decode_bf16 is None:
            return _aten_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        q_flat = query.squeeze(0).squeeze(1).contiguous()
        k_flat = key.squeeze(0).contiguous()
        v_flat = value.squeeze(0).contiguous()
        out_flat = torch.empty(Hq, D, dtype=torch.bfloat16)
        _flash_attn_decode_bf16(
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            seq_len,
            D,
            sm_scale,
            Hq,
            Hkv,
            k_flat.stride(1),
            v_flat.stride(1),
        )
        return out_flat.unsqueeze(0).unsqueeze(2)

    # Prefill fast path: Triton Flash Attention kernel (requires M >= BLOCK_M).
    force_codegen = os.getenv(
        "FLAGGEMS_ARM_ATTN_FORCE_CODEGEN", "0"
    ).lower() in {"1", "true", "on"}
    use_triton = (
        query.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and (M >= _BLOCK_M or force_codegen)
        and D in {16, 32, 64, 128, 256}
    )

    if not use_triton:
        log.debug(
            "GEMS SDPA: ATen fallback (M=%d, dtype=%s, mask=%s)",
            M,
            query.dtype,
            attn_mask is not None,
        )
        return _aten_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    sm_scale = scale if scale is not None else D**-0.5
    log.debug(
        "GEMS SDPA: Triton Flash Attention (M=%d, N=%d, D=%d, causal=%s, Hq=%d, Hkv=%d)",
        M,
        key.shape[2],
        D,
        is_causal,
        Hq,
        key.shape[1],
    )
    return _triton_flash_attn(query, key, value, sm_scale, is_causal)
