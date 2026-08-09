"""Monkey-patch Qwen3_5GatedDeltaNet.recurrent_gated_delta_rule with an
ordinary Triton decode kernel.

Replaces the per-layer ATen sequence

    state *= exp(g)
    kv_mem  = (state * k.unsqueeze(-1)).sum(dim=-2)
    delta   = (v - kv_mem) * beta
    state  += k.unsqueeze(-1) * delta.unsqueeze(-2)
    out     = (state * q.unsqueeze(-1)).sum(dim=-2)

with compiler-visible, register-bounded loops over the recurrent state.

The ordinary path uses two register-bounded state passes and removes the final
output reduction algebraically.  An optional TLE implementation remains the
automatic choice where it is installed because it is still faster for this
state-heavy operation.

Decode (T=1) only — prefill (T>1) falls back to torch_chunk_gated_delta_rule.
"""

import logging
import os

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cpu.tle_ops import (
        gated_delta_decode as _tle_gated_delta_decode,
    )
except ImportError:
    _tle_gated_delta_decode = None

from ..vector_config import ELEMENTWISE_TILE, REDUCTION_TILE

logger = logging.getLogger(__name__)

_PATCHED: set = set()
_IMPLEMENTATION = os.environ.get(
    "FLAGGEMS_ARM_GATED_DELTA_IMPL", "auto"
).lower()


@triton.jit
def _gated_delta_decode_tle_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    use_l2norm: tl.constexpr,
):
    _tle_gated_delta_decode(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        state_ptr,
        out_ptr,
        B,
        H,
        k_dim,
        v_dim,
        use_l2norm,
    )


@triton.jit
def _gated_delta_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    use_l2norm: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    v_blocks = tl.cdiv(v_dim, BLOCK_V)
    pid = tl.program_id(0)
    head = pid // v_blocks
    v_block = pid % v_blocks
    v_lanes = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_lanes < v_dim

    q_base = head * k_dim
    k_base = head * k_dim
    value_base = head * v_dim
    state_base = head * k_dim * v_dim

    q_scale = 1.0 / tl.sqrt(float(k_dim))
    k_scale = 1.0
    if use_l2norm:
        k_lanes = tl.arange(0, BLOCK_K)
        q_sum_sq = tl.zeros((1,), tl.float32)
        k_sum_sq = tl.zeros((1,), tl.float32)
        for base in tl.range(0, k_dim, BLOCK_K, loop_unroll_factor=1):
            indices = base + k_lanes
            mask = indices < k_dim
            q_values = tl.load(
                q_ptr + q_base + indices, mask=mask, other=0.0
            ).to(tl.float32)
            k_values = tl.load(
                k_ptr + k_base + indices, mask=mask, other=0.0
            ).to(tl.float32)
            q_sum_sq += tl.sum(q_values * q_values, axis=0)
            k_sum_sq += tl.sum(k_values * k_values, axis=0)
        q_scale *= 1.0 / tl.sqrt(q_sum_sq + 1.0e-6)
        k_scale = 1.0 / tl.sqrt(k_sum_sq + 1.0e-6)

    decay = tl.exp(tl.load(g_ptr + head).to(tl.float32))
    beta = tl.load(beta_ptr + head).to(tl.float32)
    kv_memory = tl.zeros((BLOCK_V,), tl.float32)
    output_base = tl.zeros((BLOCK_V,), tl.float32)
    kq_dot = tl.zeros((1,), tl.float32)

    # First pass computes both state projections.  Keeping only two output
    # vectors live avoids materializing the k_dim x BLOCK_V tile in SSA.
    for k_index in tl.range(0, k_dim, loop_unroll_factor=4):
        q = tl.load(q_ptr + q_base + k_index).to(tl.float32) * q_scale
        k = tl.load(k_ptr + k_base + k_index).to(tl.float32) * k_scale
        state = tl.load(
            state_ptr + state_base + k_index * v_dim + v_lanes,
            mask=v_mask,
            other=0.0,
        )
        decayed_state = state * decay
        kv_memory += decayed_state * k
        output_base += decayed_state * q
        kq_dot += k * q

    value = tl.load(
        v_ptr + value_base + v_lanes, mask=v_mask, other=0.0
    ).to(tl.float32)
    delta = (value - kv_memory) * beta

    # The rank-one update contributes delta * dot(k, q) to the output, so a
    # second output reduction is unnecessary.  The second state pass exists
    # only to write the updated cache.
    for k_index in tl.range(0, k_dim, loop_unroll_factor=4):
        k = tl.load(k_ptr + k_base + k_index).to(tl.float32) * k_scale
        state_address = state_ptr + state_base + k_index * v_dim + v_lanes
        state = tl.load(state_address, mask=v_mask, other=0.0)
        tl.store(state_address, state * decay + k * delta, mask=v_mask)

    tl.store(
        out_ptr + value_base + v_lanes,
        output_base + kq_dot * delta,
        mask=v_mask,
    )


def _patched_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """Drop-in for torch_recurrent_gated_delta_rule on T=1 decode path.

    Shapes (matching HF):
      query, key:    [B, T, H, k_dim] (any dtype; cast to fp32 internally)
      value:         [B, T, H, v_dim]
      g, beta:       [B, T, H]
      initial_state: [B, H, k_dim, v_dim] fp32, or None

    Returns:
      core_attn_out:        [B, T, H, v_dim] cast back to query.dtype
      last_recurrent_state: [B, H, k_dim, v_dim] fp32 (or None)

    For T>1, falls back to the original torch implementation in the host
    module (caller passes that in via the closure during patching).
    """
    raise NotImplementedError("install via patch_qwen3_5_gated_delta(model)")


def _make_patched_fn(torch_recurrent_fn):
    def fn(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel=False,
    ):
        B, T, H, k_dim = query.shape
        v_dim = value.shape[-1]

        # Prefill or any non-decode shape: defer to the torch reference.
        if T != 1 or k_dim > 256 or v_dim > 256 or k_dim % 4 != 0 or v_dim % 4 != 0:
            return torch_recurrent_fn(
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
            )

        orig_dtype = query.dtype

        use_tle = _tle_gated_delta_decode is not None and _IMPLEMENTATION in (
            "auto",
            "tle",
        )

        # T=1 model views are contiguous after squeezing.
        q_f = query.squeeze(1).contiguous()
        k_f = key.squeeze(1).contiguous()
        v_f = value.squeeze(1).contiguous()
        g_f = g.squeeze(1).contiguous()
        b_f = beta.squeeze(1).contiguous()

        if initial_state is None:
            state = torch.zeros(B, H, k_dim, v_dim, dtype=torch.float32)
        else:
            # .contiguous() on already-contiguous fp32 is a no-op.
            # The caller replaces cache_params.recurrent_states[layer_idx]
            # with our return value, so in-place update is safe here.
            state = initial_state.to(torch.float32).contiguous()

        if use_tle:
            # The legacy runtime ABI consumes FP32 inputs and output.  Keep it
            # as the automatic choice where available: the audit found its
            # recurrent-state loop still faster than current ordinary codegen.
            q_runtime = q_f.to(torch.float32)
            k_runtime = k_f.to(torch.float32)
            v_runtime = v_f.to(torch.float32)
            g_runtime = g_f.to(torch.float32)
            b_runtime = b_f.to(torch.float32)
            out = torch.empty(B, H, v_dim, dtype=torch.float32)
            _gated_delta_decode_tle_kernel[(1,)](
                q_runtime,
                k_runtime,
                v_runtime,
                g_runtime,
                b_runtime,
                state,
                out,
                B=B,
                H=H,
                k_dim=k_dim,
                v_dim=v_dim,
                use_l2norm=1 if use_qk_l2norm_in_kernel else 0,
            )
            core_attn_out = out.unsqueeze(1).to(orig_dtype).contiguous()
        else:
            # Stock Triton-CPU fallback: direct BF16 loads avoid the temporary
            # FP32 tensors used by the runtime ABI.
            out = torch.empty(B, H, v_dim, dtype=orig_dtype)
            grid = (B * H * triton.cdiv(v_dim, ELEMENTWISE_TILE),)
            _gated_delta_decode_kernel[grid](
                q_f,
                k_f,
                v_f,
                g_f,
                b_f,
                state,
                out,
                B=B,
                H=H,
                k_dim=k_dim,
                v_dim=v_dim,
                use_l2norm=1 if use_qk_l2norm_in_kernel else 0,
                BLOCK_K=REDUCTION_TILE,
                BLOCK_V=ELEMENTWISE_TILE,
                num_warps=1,
                num_stages=1,
            )
            core_attn_out = out.unsqueeze(1).contiguous()
        last_recurrent_state = state if output_final_state else None
        return core_attn_out, last_recurrent_state

    return fn


def _get_qwen3_5_gated_delta_classes() -> tuple:
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5GatedDeltaNet"),
        (
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            "Qwen3_5MoeGatedDeltaNet",
        ),
        (
            "transformers.models.qwen3_next.modeling_qwen3_next",
            "Qwen3NextGatedDeltaNet",
        ),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_gated_delta(model) -> int:
    """Replace each GDN module's recurrent rule with the audited auto path.

    Returns the number of modules patched.

    Safe to call multiple times (each module is patched once via id-tracking).
    """
    gdn_classes = _get_qwen3_5_gated_delta_classes()
    if not gdn_classes:
        logger.debug("No Qwen GDN classes found in transformers, skipping patch")
        return 0

    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) not in _PATCHED:
            torch_recurrent_fn = module.recurrent_gated_delta_rule
            module._original_recurrent_gated_delta_rule = torch_recurrent_fn
            module.recurrent_gated_delta_rule = _make_patched_fn(torch_recurrent_fn)
            _PATCHED.add(id(module))
            n += 1
    if n > 0:
        cls_names = ", ".join(c.__name__ for c in gdn_classes)
        logger.info(
            "Patched %d GDN modules (classes: %s) with gated-delta auto path",
            n,
            cls_names,
        )
    return n


def unpatch_qwen3_5_gated_delta(model) -> int:
    gdn_classes = _get_qwen3_5_gated_delta_classes()
    if not gdn_classes:
        return 0
    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) in _PATCHED:
            if hasattr(module, "_original_recurrent_gated_delta_rule"):
                module.recurrent_gated_delta_rule = (
                    module._original_recurrent_gated_delta_rule
                )
                del module._original_recurrent_gated_delta_rule
            _PATCHED.discard(id(module))
            n += 1
    return n
