"""Fused Triton CPU kernels for gated-delta decode."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_conv1d_silu_chunk(
    mixed_ptr,
    conv_state_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    slot,
    channels,
    CONV_STATE_STRIDE_0: tl.constexpr,
    CONV_STATE_STRIDE_1: tl.constexpr,
    CONV_STATE_STRIDE_2: tl.constexpr,
    CONV_WEIGHT_STRIDE_0: tl.constexpr,
    CONV_WEIGHT_STRIDE_1: tl.constexpr,
):
    """Width-four depthwise causal convolution, updating state in place."""
    token = tl.load(mixed_ptr + channels).to(tl.float32)
    state_base = (
        slot * CONV_STATE_STRIDE_0 + channels * CONV_STATE_STRIDE_1
    )
    state0 = tl.load(conv_state_ptr + state_base).to(tl.float32)
    state1 = tl.load(
        conv_state_ptr + state_base + CONV_STATE_STRIDE_2
    ).to(tl.float32)
    state2 = tl.load(
        conv_state_ptr + state_base + 2 * CONV_STATE_STRIDE_2
    ).to(tl.float32)
    weight_base = channels * CONV_WEIGHT_STRIDE_0
    weight0 = tl.load(conv_weight_ptr + weight_base).to(tl.float32)
    weight1 = tl.load(
        conv_weight_ptr + weight_base + CONV_WEIGHT_STRIDE_1
    ).to(tl.float32)
    weight2 = tl.load(
        conv_weight_ptr + weight_base + 2 * CONV_WEIGHT_STRIDE_1
    ).to(tl.float32)
    weight3 = tl.load(
        conv_weight_ptr + weight_base + 3 * CONV_WEIGHT_STRIDE_1
    ).to(tl.float32)
    result = (
        state0 * weight0
        + state1 * weight1
        + state2 * weight2
        + token * weight3
        + tl.load(conv_bias_ptr + channels).to(tl.float32)
    )
    result *= 1.0 / (1.0 + tl.exp(-result))
    rounded = result.to(tl.bfloat16)
    tl.store(conv_state_ptr + state_base, state1)
    tl.store(
        conv_state_ptr + state_base + CONV_STATE_STRIDE_2, state2
    )
    tl.store(
        conv_state_ptr + state_base + 2 * CONV_STATE_STRIDE_2, token
    )
    tl.store(mixed_ptr + channels, rounded)
    return rounded.to(tl.float32)


@triton.jit
def _gdn_packed_decode_kernel(
    mixed_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    conv_state_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    recurrent_state_ptr,
    state_indices_ptr,
    output_ptr,
    KEY_HEADS: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    CONV_STATE_STRIDE_0: tl.constexpr,
    CONV_STATE_STRIDE_1: tl.constexpr,
    CONV_STATE_STRIDE_2: tl.constexpr,
    CONV_WEIGHT_STRIDE_0: tl.constexpr,
    CONV_WEIGHT_STRIDE_1: tl.constexpr,
    QUERY_POST_SCALE: tl.constexpr,
):
    """One launch fuses Conv1D and recurrence for one key-head group."""
    key_head = tl.program_id(0)
    slot = tl.load(state_indices_ptr).to(tl.int64)
    lanes = tl.arange(0, KEY_DIM)
    q_width: tl.constexpr = KEY_HEADS * KEY_DIM
    value_base: tl.constexpr = 2 * q_width
    query_channels = key_head * KEY_DIM + lanes
    key_channels = q_width + key_head * KEY_DIM + lanes
    query = _gdn_conv1d_silu_chunk(
        mixed_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        conv_bias_ptr,
        slot,
        query_channels,
        CONV_STATE_STRIDE_0,
        CONV_STATE_STRIDE_1,
        CONV_STATE_STRIDE_2,
        CONV_WEIGHT_STRIDE_0,
        CONV_WEIGHT_STRIDE_1,
    )
    key = _gdn_conv1d_silu_chunk(
        mixed_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        conv_bias_ptr,
        slot,
        key_channels,
        CONV_STATE_STRIDE_0,
        CONV_STATE_STRIDE_1,
        CONV_STATE_STRIDE_2,
        CONV_WEIGHT_STRIDE_0,
        CONV_WEIGHT_STRIDE_1,
    )
    query_scale = (
        tl.rsqrt(tl.sum(query * query, axis=0) + 1.0e-6)
        * QUERY_POST_SCALE
    )
    key_scale = tl.rsqrt(tl.sum(key * key, axis=0) + 1.0e-6)
    head_group: tl.constexpr = VALUE_HEADS // KEY_HEADS
    key_lanes = tl.arange(0, BLOCK_KEY)

    for group in range(0, head_group):
        value_head = key_head * head_group + group
        value_channels = value_base + value_head * VALUE_DIM + lanes
        _gdn_conv1d_silu_chunk(
            mixed_ptr,
            conv_state_ptr,
            conv_weight_ptr,
            conv_bias_ptr,
            slot,
            value_channels,
            CONV_STATE_STRIDE_0,
            CONV_STATE_STRIDE_1,
            CONV_STATE_STRIDE_2,
            CONV_WEIGHT_STRIDE_0,
            CONV_WEIGHT_STRIDE_1,
        )
        a_value = tl.load(a_ptr + value_head).to(tl.float32)
        b_value = tl.load(b_ptr + value_head).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)
        a_log = tl.load(a_log_ptr + value_head)
        softplus_input = a_value + dt_bias
        softplus = tl.where(
            softplus_input > 20.0,
            softplus_input,
            tl.log(1.0 + tl.exp(softplus_input)),
        )
        decay = tl.exp(-tl.exp(a_log) * softplus)
        beta = 1.0 / (1.0 + tl.exp(-b_value))
        state_head_offset = (
            slot * VALUE_HEADS * VALUE_DIM * KEY_DIM
            + value_head * VALUE_DIM * KEY_DIM
        )
        for value_index in tl.range(
            0, VALUE_DIM, 1, loop_unroll_factor=1
        ):
            predicted = 0.0
            for key_start in tl.range(
                0, KEY_DIM, BLOCK_KEY, loop_unroll_factor=1
            ):
                key_index = key_start + key_lanes
                state_chunk = tl.load(
                    recurrent_state_ptr
                    + state_head_offset
                    + value_index * KEY_DIM
                    + key_index
                )
                key_chunk = tl.load(
                    mixed_ptr + q_width + key_head * KEY_DIM + key_index
                ).to(tl.float32)
                predicted += tl.sum(
                    state_chunk * decay * key_chunk * key_scale, axis=0
                )
            value = tl.load(
                mixed_ptr + value_base + value_head * VALUE_DIM + value_index
            ).to(tl.float32)
            delta = (value - predicted) * beta
            result = 0.0
            for key_start in tl.range(
                0, KEY_DIM, BLOCK_KEY, loop_unroll_factor=1
            ):
                key_index = key_start + key_lanes
                state_offset = (
                    state_head_offset + value_index * KEY_DIM + key_index
                )
                state_chunk = tl.load(recurrent_state_ptr + state_offset)
                key_chunk = tl.load(
                    mixed_ptr + q_width + key_head * KEY_DIM + key_index
                ).to(tl.float32)
                query_chunk = tl.load(
                    mixed_ptr + key_head * KEY_DIM + key_index
                ).to(tl.float32)
                updated_state = (
                    state_chunk * decay + key_chunk * key_scale * delta
                )
                tl.store(recurrent_state_ptr + state_offset, updated_state)
                result += tl.sum(
                    updated_state * query_chunk * query_scale, axis=0
                )
            tl.store(
                output_ptr + value_head * VALUE_DIM + value_index, result
            )


def gdn_packed_decode_triton_out(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    output: torch.Tensor,
    block_key: int = 0,
    cpu_threads: int = 0,
) -> None:
    """Fused single-token packed Conv1D + GDN recurrence."""
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[0] != 1:
        raise ValueError("packed Triton GDN decode currently requires one token")
    if state_indices.numel() != 1 or state_indices.dtype != torch.int32:
        raise ValueError("packed Triton GDN decode requires one INT32 state index")
    if recurrent_state.ndim != 4 or recurrent_state.dtype != torch.float32:
        raise ValueError("recurrent state must be FP32 [slots,H,V,K]")
    value_heads, value_dim, key_dim = recurrent_state.shape[1:]
    if value_dim != key_dim or key_dim <= 0 or key_dim & (key_dim - 1):
        raise ValueError("packed Triton GDN decode requires equal power-of-two V/K")
    value_width = value_heads * value_dim
    qk_width = mixed_qkv.shape[1] - value_width
    if qk_width <= 0 or qk_width % (2 * key_dim):
        raise ValueError("cannot infer packed GDN key-head count")
    key_heads = qk_width // (2 * key_dim)
    if value_heads % key_heads:
        raise ValueError("value heads must be divisible by key heads")
    conv_dim = mixed_qkv.shape[1]
    if (
        mixed_qkv.dtype != torch.bfloat16
        or not mixed_qkv.is_contiguous()
        or conv_state.dtype != torch.bfloat16
        or conv_state.ndim != 3
        or conv_state.shape[1:] != (conv_dim, 3)
        or conv_weight.dtype != torch.bfloat16
        or conv_weight.shape != (conv_dim, 4)
        or conv_bias.dtype != torch.bfloat16
        or conv_bias.shape != (conv_dim,)
        or a.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or dt_bias.dtype != torch.bfloat16
        or a.numel() != value_heads
        or b.numel() != value_heads
        or dt_bias.numel() != value_heads
        or a_log.dtype != torch.float32
        or a_log.numel() != value_heads
        or output.dtype != torch.bfloat16
        or output.numel() != value_width
    ):
        raise ValueError("unsupported packed GDN tensor layout or dtype")
    if block_key == 0:
        block_key = next(
            candidate
            for candidate in (32, 16, 8, 4, 2, 1)
            if key_dim % candidate == 0
        )
    if block_key <= 0 or key_dim % block_key or block_key & (block_key - 1):
        raise ValueError("block_key must be a power-of-two divisor of key_dim")
    _gdn_packed_decode_kernel[(key_heads,)](
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        conv_state,
        conv_weight,
        conv_bias,
        recurrent_state,
        state_indices,
        output,
        KEY_HEADS=key_heads,
        VALUE_HEADS=value_heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_KEY=block_key,
        CONV_STATE_STRIDE_0=conv_state.stride(0),
        CONV_STATE_STRIDE_1=conv_state.stride(1),
        CONV_STATE_STRIDE_2=conv_state.stride(2),
        CONV_WEIGHT_STRIDE_0=conv_weight.stride(0),
        CONV_WEIGHT_STRIDE_1=conv_weight.stride(1),
        QUERY_POST_SCALE=1.0 / math.sqrt(key_dim),
        num_warps=1,
        num_stages=1,
        num_cpu_threads=cpu_threads,
    )
