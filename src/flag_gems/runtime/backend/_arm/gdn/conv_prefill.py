# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Existing compiler-visible Arm kernels; native packed ABI, ordinary Triton math."""

import ctypes
import threading
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _round_to_nearest_even_i32(value):
    return libdevice.rint(value).to(tl.int32)


@triton.jit
def _sleef_expf_u10_inline(value):
    """SLEEF-u10 exp polynomial expressed as ordinary Triton arithmetic."""
    exponent = _round_to_nearest_even_i32(value * 1.4426950408889634)
    exponent_f = exponent.to(tl.float32)
    reduced = tl.fma(exponent_f, -0.693145751953125, value)
    reduced = tl.fma(exponent_f, -1.428606765330187e-6, reduced)

    polynomial = tl.full(value.shape, 0.00019852761761285365, tl.float32)
    polynomial = tl.fma(polynomial, reduced, 0.0013930435525253415)
    polynomial = tl.fma(polynomial, reduced, 0.008333360776305199)
    polynomial = tl.fma(polynomial, reduced, 0.041666485369205475)
    polynomial = tl.fma(polynomial, reduced, 0.1666666716337204)
    polynomial = tl.fma(polynomial, reduced, 0.5)
    result = 1.0 + tl.fma(reduced * reduced, polynomial, reduced)

    half_exponent = exponent >> 1
    pow0_bits = (half_exponent + 127) << 23
    pow1_bits = (exponent - half_exponent + 127) << 23
    pow0 = pow0_bits.to(tl.float32, bitcast=True)
    pow1 = pow1_bits.to(tl.float32, bitcast=True)
    result = result * pow0 * pow1
    result = tl.where(value < -104.0, 0.0, result)
    return tl.where(value > 100.0, float("inf"), result)


@triton.jit(
    do_not_specialize=[
        "input_stride0",
        "input_stride1",
        "state_stride0",
        "state_stride1",
        "state_stride2",
        "weight_stride0",
        "weight_stride1",
        "output_stride0",
        "output_stride1",
    ],
    do_not_specialize_on_alignment=[
        "input_ptr",
        "state_ptr",
        "weight_ptr",
        "bias_ptr",
        "output_ptr",
        "starts_ptr",
        "slots_ptr",
        "initial_ptr",
    ],
)
def _gdn_conv1d_prefill_width4_bf16_kernel(
    input_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    starts_ptr,
    slots_ptr,
    initial_ptr,
    input_stride0,
    input_stride1,
    state_stride0,
    state_stride1,
    state_stride2,
    weight_stride0,
    weight_stride1,
    output_stride0,
    output_stride1,
    DIM: tl.constexpr,
    CHANNEL_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
):
    """Vector-channel GDN depthwise causal conv for vLLM prefill."""
    work = tl.program_id(0)
    sequence = work // CHANNEL_BLOCKS
    channel_block = work % CHANNEL_BLOCKS
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = channels < DIM
    token_begin = tl.load(starts_ptr + sequence)
    token_end = tl.load(starts_ptr + sequence + 1)
    slot = tl.load(slots_ptr + sequence)
    has_initial = tl.load(initial_ptr + sequence)
    state_base = state_ptr + slot * state_stride0 + channels * state_stride1
    history0 = tl.where(
        has_initial,
        tl.load(state_base, mask=mask, other=0.0).to(tl.float32),
        0.0,
    )
    history1 = tl.where(
        has_initial,
        tl.load(state_base + state_stride2, mask=mask, other=0.0).to(tl.float32),
        0.0,
    )
    history2 = tl.where(
        has_initial,
        tl.load(state_base + 2 * state_stride2, mask=mask, other=0.0).to(tl.float32),
        0.0,
    )
    weight_base = weight_ptr + channels * weight_stride0
    weight0 = tl.load(weight_base, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight_base + weight_stride1, mask=mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight_base + 2 * weight_stride1, mask=mask, other=0.0).to(
        tl.float32
    )
    weight3 = tl.load(weight_base + 3 * weight_stride1, mask=mask, other=0.0).to(
        tl.float32
    )
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for token in tl.range(token_begin, token_end):
        value = tl.load(
            input_ptr + channels * input_stride0 + token * input_stride1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        result = bias + history0 * weight0
        result += history1 * weight1
        result += history2 * weight2
        result += value * weight3
        if SILU:
            result = result / (1.0 + _sleef_expf_u10_inline(-result))
        tl.store(
            output_ptr + channels * output_stride0 + token * output_stride1,
            result,
            mask=mask,
        )
        history0 = history1
        history1 = history2
        history2 = value

    # A new request clears its causal history even when its token range is
    # empty. Existing requests retain history; rollback positions are untouched.
    tl.store(state_base, history0, mask=mask)
    tl.store(state_base + state_stride2, history1, mask=mask)
    tl.store(state_base + 2 * state_stride2, history2, mask=mask)


# Compilation and module ownership remain outside the parallel compute path.

_kernel_lock = threading.Lock()
_kernel_owners = {}


def prepare_conv_prefill(library_path, dim, has_bias=True, silu=True):
    key = (str(library_path), dim, has_bias, silu)
    if key in _kernel_owners:
        return
    with _kernel_lock:
        if key in _kernel_owners:
            return
        native = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        data = torch.empty(1, dtype=torch.bfloat16, device="cpu")
        metadata = torch.empty(1, dtype=torch.int32, device="cpu")
        flags = torch.empty(1, dtype=torch.bool, device="cpu")
        compiled = _gdn_conv1d_prefill_width4_bf16_kernel.warmup(
            data,
            data,
            data,
            data,
            data,
            metadata,
            metadata,
            flags,
            *([13] * 9),
            DIM=dim,
            CHANNEL_BLOCKS=(dim + 63) // 64,
            BLOCK_C=64,
            HAS_BIAS=has_bias,
            SILU=silu,
            grid=(1,),
            enable_fp_fusion=False,
        )
        dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
        address = ctypes.cast(getattr(dso, compiled.name), ctypes.c_void_p).value
        register = native.flag_gems_arm_register_conv_kernel
        register.argtypes = [
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_uint64,
        ]
        register.restype = ctypes.c_int32
        if register(dim, has_bias, silu, address):
            raise RuntimeError("Cannot register the CPU Triton conv kernel")
        _kernel_owners[key] = (native, compiled, dso)


def gdn_conv1d_prefill_triton(
    input,
    conv_state,
    weight,
    bias,
    query_start_loc,
    cache_indices,
    has_initial_state,
    silu_activation=True,
):
    from ..fused.native_ops import arm_native_ops

    ops = arm_native_ops("gdn")
    # Read the initialized owner after the lazy loader has returned.
    from ..fused.native_ops import _library

    prepare_conv_prefill(_library[0], input.shape[0], bias is not None, silu_activation)
    return ops.gdn_conv1d_prefill.triton(
        input,
        conv_state,
        weight,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
    )


@lru_cache(maxsize=64)
def _update_metadata(tokens):
    # Each decode token owns one independent cache slot. These read-only CPU
    # tensors are shared across calls; avoid allocating metadata per decode.
    return (
        torch.arange(tokens + 1, dtype=torch.int32, device="cpu"),
        torch.ones(tokens, dtype=torch.bool, device="cpu"),
    )


def gdn_conv1d_update_triton(
    input,
    conv_state,
    weight,
    bias,
    state_indices,
    silu_activation=True,
):
    """Width-four BF16 decode conv, reusing the ordinary Triton prefill kernel.

    Input is [tokens, dim], history is [slots, dim, 3], and state_indices
    selects one distinct slot per token. State is updated in place. BF16 SiLU
    outputs may differ from scalar expf by one BF16 step; history is exact.
    """
    if (
        input.ndim != 2
        or conv_state.ndim != 3
        or conv_state.shape[2] != 3
        or weight.ndim != 2
        or weight.shape[1] != 4
    ):
        raise ValueError("Triton conv update requires width 4 and history length 3")
    if state_indices.ndim != 1 or state_indices.numel() != input.shape[0]:
        raise ValueError("Triton conv update needs one state index per token")
    starts, initial = _update_metadata(input.shape[0])
    return gdn_conv1d_prefill_triton(
        input.t(),
        conv_state,
        weight,
        bias,
        starts,
        state_indices,
        initial,
        silu_activation,
    ).t()
