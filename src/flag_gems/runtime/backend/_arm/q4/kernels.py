"""Ordinary-Triton kernels for the ARM Q4A8 production path.

The matrix kernel consumes the native KAI qsi8d32p/qsi4c32p physical ABI,
but the computation remains visible as ``tl.dot``. Triton-CPU recognizes the
layout algebra and lowers the dot to fixed-width NEON I8MM. There is no TLE
runtime call in the matrix kernel.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..ops.silu_and_mul import _sleef_expf_u10_inline


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
    ]
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
    state_base = (
        state_ptr + slot * state_stride0 + channels * state_stride1
    )
    history0 = tl.where(
        has_initial,
        tl.load(state_base, mask=mask, other=0.0).to(tl.float32),
        0.0,
    )
    history1 = tl.where(
        has_initial,
        tl.load(
            state_base + state_stride2, mask=mask, other=0.0
        ).to(tl.float32),
        0.0,
    )
    history2 = tl.where(
        has_initial,
        tl.load(
            state_base + 2 * state_stride2, mask=mask, other=0.0
        ).to(tl.float32),
        0.0,
    )
    weight_base = weight_ptr + channels * weight_stride0
    weight0 = tl.load(weight_base, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(
        weight_base + weight_stride1, mask=mask, other=0.0
    ).to(tl.float32)
    weight2 = tl.load(
        weight_base + 2 * weight_stride1, mask=mask, other=0.0
    ).to(tl.float32)
    weight3 = tl.load(
        weight_base + 3 * weight_stride1, mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for token in tl.range(token_begin, token_end):
        value = tl.load(
            input_ptr
            + channels * input_stride0
            + token * input_stride1,
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
            output_ptr
            + channels * output_stride0
            + token * output_stride1,
            result,
            mask=mask,
        )
        history0 = history1
        history1 = history2
        history2 = value

    tl.store(state_base, history0, mask=mask)
    tl.store(state_base + state_stride2, history1, mask=mask)
    tl.store(state_base + 2 * state_stride2, history2, mask=mask)


@triton.jit
def _round_to_nearest_even(value):
    """Standard RNE; LLVM selects the target's native rounding instruction."""
    return libdevice.rint(value)


@triton.jit
def _quantize_symmetric_i8(values, inv_scale):
    scaled = tl.minimum(
        tl.maximum(values * inv_scale, -127.0), 127.0
    )
    return _round_to_nearest_even(scaled).to(tl.int8)


@triton.jit
def _quantize_symmetric_i8_from_absmax(values, inv_scale):
    """Skip saturation when inv_scale comes from this K32 block's absmax."""
    return _round_to_nearest_even(values * inv_scale).to(tl.int8)


@triton.jit
def _quantize_token_asymmetric_i8(values, inv_scale, zero_point):
    """Match compressed-tensors BF16 token-wise asymmetric fake quantization."""
    # The source checkpoint computes x / scale and the zero-point addition in
    # BF16 before RNE.  Preserve those two rounding boundaries explicitly.
    scaled = (values.to(tl.float32) * inv_scale).to(tl.bfloat16)
    shifted = (
        scaled.to(tl.float32) + zero_point.to(tl.float32)
    ).to(tl.bfloat16)
    clipped = tl.minimum(
        tl.maximum(shifted.to(tl.float32), -128.0), 127.0
    )
    return _round_to_nearest_even(clipped).to(tl.int8)


@triton.jit
def _round_away_from_zero(value):
    """Match KAI/AArch64 ties-away rounding as a compiler-visible math op."""
    return libdevice.round(value.to(tl.float32)).to(tl.int32)


@triton.jit
def _quantize_kai_asymmetric_i8(values, quant_multiplier, zero_point):
    rounded = _round_away_from_zero(
        values.to(tl.float32) * quant_multiplier
    )
    shifted = rounded + zero_point.to(tl.int32)
    return tl.minimum(tl.maximum(shifted, -128), 127).to(tl.int8)


@triton.jit
def _q4_kai_asymmetric_qparams_from_minmax(row_min, row_max):
    """KleidiAI ``qai8dxp_f32`` row quantization parameters."""
    row_min = tl.minimum(row_min, 0.0)
    row_max = tl.maximum(row_max, 0.0)
    value_range = row_max - row_min
    quant_multiplier = tl.where(value_range == 0.0, 1.0, 255.0 / value_range)
    dequant_scale = tl.where(
        quant_multiplier == 0.0, 0.0, 1.0 / quant_multiplier
    )
    descaled_min = row_min * quant_multiplier
    descaled_max = row_max * quant_multiplier
    choose_min = -128.0 + descaled_min + 127.0 + descaled_max > 0.0
    zero_point = tl.where(
        choose_min, -128.0 - descaled_min, 127.0 - descaled_max
    )
    zero_point = tl.minimum(tl.maximum(zero_point, -128.0), 127.0)
    zero_point = _round_to_nearest_even(zero_point).to(tl.int8)
    return dequant_scale, quant_multiplier, zero_point


@triton.jit
def _q4_token_asymmetric_qparams_kai_f32(x_base, K: tl.constexpr):
    lanes = tl.arange(0, 16)
    row_min = tl.full((1,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((1,), -3.4028234663852886e38, tl.float32)
    for start in tl.range(0, K, 16, loop_unroll_factor=1):
        values = tl.load(x_base + start + lanes).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values, axis=0))
        row_max = tl.maximum(row_max, tl.max(values, axis=0))
    return _q4_kai_asymmetric_qparams_from_minmax(row_min, row_max)


@triton.jit
def _q4_store_token_asymmetric_k32_kai(
    data_base, x_base, quant_multiplier, zero_point
):
    lanes = tl.arange(0, 8)
    quant0 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + lanes), quant_multiplier, zero_point
    )
    quant1 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + 8 + lanes), quant_multiplier, zero_point
    )
    quant2 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + 16 + lanes), quant_multiplier, zero_point
    )
    quant3 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + 24 + lanes), quant_multiplier, zero_point
    )
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
    quantized = tl.join(quant01, quant23).permute(1, 0).reshape((32,))
    tl.store(data_base + tl.arange(0, 32), quantized)


@triton.jit
def _q4_asymmetric_qparams_from_minmax(row_min, row_max):
    row_min = tl.minimum(row_min, 0.0).to(tl.bfloat16)
    row_max = tl.maximum(row_max, 0.0).to(tl.bfloat16)
    # compressed-tensors inherits the BF16 source dtype for scale arithmetic.
    scale = (
        (row_max.to(tl.float32) - row_min.to(tl.float32)) / 255.0
    ).to(tl.bfloat16)
    # compressed-tensors' _get_dtype_eps uses BF16 epsilon for an all-zero
    # token.  This branch does not affect ordinary model activations, but its
    # exact value is part of the documented frontend contract.
    scale = tl.where(
        scale.to(tl.float32) == 0.0,
        0.0078125,
        scale.to(tl.float32),
    ).to(tl.bfloat16)
    min_over_scale = (
        row_min.to(tl.float32) / scale.to(tl.float32)
    ).to(tl.bfloat16)
    zp_value = (
        -128.0 - min_over_scale.to(tl.float32)
    ).to(tl.bfloat16)
    zp_value = tl.minimum(
        tl.maximum(zp_value.to(tl.float32), -128.0), 127.0
    )
    zero_point = _round_to_nearest_even(zp_value).to(tl.int8)
    inv_scale = 1.0 / scale.to(tl.float32)
    return scale, inv_scale, zero_point


@triton.jit
def _q4_token_asymmetric_qparams_bf16(x_base, K: tl.constexpr):
    """Return exact BF16 scale/int8 zp for one finite token row."""
    lanes = tl.arange(0, 16)
    row_min = tl.full((1,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((1,), -3.4028234663852886e38, tl.float32)
    for start in tl.range(0, K, 16, loop_unroll_factor=1):
        values = tl.load(x_base + start + lanes).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values, axis=0))
        row_max = tl.maximum(row_max, tl.max(values, axis=0))
    return _q4_asymmetric_qparams_from_minmax(row_min, row_max)


@triton.jit
def _q4_store_token_asymmetric_k32(
    data_base, x_base, inv_scale, zero_point
):
    """Quantize one K32 slice into the compact asymmetric decode ABI."""
    lanes = tl.arange(0, 8)
    quant0 = _quantize_token_asymmetric_i8(
        tl.load(x_base + lanes), inv_scale, zero_point
    )
    quant1 = _quantize_token_asymmetric_i8(
        tl.load(x_base + 8 + lanes), inv_scale, zero_point
    )
    quant2 = _quantize_token_asymmetric_i8(
        tl.load(x_base + 16 + lanes), inv_scale, zero_point
    )
    quant3 = _quantize_token_asymmetric_i8(
        tl.load(x_base + 24 + lanes), inv_scale, zero_point
    )
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
    quantized = tl.join(quant01, quant23).permute(1, 0).reshape((32,))
    tl.store(data_base + tl.arange(0, 32), quantized)


@triton.jit
def _q4_store_token_asymmetric_bf16_values(
    data_base, values0, values1, values2, values3, inv_scale, zero_point
):
    """Pack four already-materialized BF16 K8 values with token qparams."""
    quant0 = _quantize_token_asymmetric_i8(values0, inv_scale, zero_point)
    quant1 = _quantize_token_asymmetric_i8(values1, inv_scale, zero_point)
    quant2 = _quantize_token_asymmetric_i8(values2, inv_scale, zero_point)
    quant3 = _quantize_token_asymmetric_i8(values3, inv_scale, zero_point)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
    quantized = tl.join(quant01, quant23).permute(1, 0).reshape((32,))
    tl.store(data_base + tl.arange(0, 32), quantized)


@triton.jit
def _q4_bf16_values_minmax(values0, values1, values2, values3):
    minimum = tl.minimum(
        tl.minimum(tl.min(values0.to(tl.float32), axis=0),
                   tl.min(values1.to(tl.float32), axis=0)),
        tl.minimum(tl.min(values2.to(tl.float32), axis=0),
                   tl.min(values3.to(tl.float32), axis=0)),
    )
    maximum = tl.maximum(
        tl.maximum(tl.max(values0.to(tl.float32), axis=0),
                   tl.max(values1.to(tl.float32), axis=0)),
        tl.maximum(tl.max(values2.to(tl.float32), axis=0),
                   tl.max(values3.to(tl.float32), axis=0)),
    )
    return minimum, maximum


@triton.jit
def _pack_lhs_qsi8d32p_row_kernel(
    x_ptr,
    lhs_scale_ptr,
    lhs_data_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Lower-register-pressure variant: one program owns one logical row."""
    row = tl.program_id(0)
    row_ok = row < M
    source_row = tl.minimum(row, M - 1)
    panel = row // 4
    panel_row = row % 4
    lanes = tl.arange(0, 8)
    groups: tl.constexpr = K // 32

    for group in tl.range(0, groups, loop_unroll_factor=1):
        # KAI stores one row as four separate K8 segments.  A vector<32xf32>
        # plus a scatter-style store keeps too many values live on AArch64
        # and spills vector registers.  Keep four native K8 slices live,
        # reduce them to one scale, then retire each slice after its contiguous
        # store.  This avoids both the scatter and a second activation load.
        x_base = x_ptr + source_row * stride_xm + group * 32
        values0 = tl.load(x_base + lanes).to(tl.float32)
        values1 = tl.load(x_base + 8 + lanes).to(tl.float32)
        values2 = tl.load(x_base + 16 + lanes).to(tl.float32)
        values3 = tl.load(x_base + 24 + lanes).to(tl.float32)
        values0 = tl.where(row_ok, values0, 0.0)
        values1 = tl.where(row_ok, values1, 0.0)
        values2 = tl.where(row_ok, values2, 0.0)
        values3 = tl.where(row_ok, values3, 0.0)
        absmax = tl.maximum(
            tl.maximum(
                tl.max(tl.abs(values0), axis=0),
                tl.max(tl.abs(values1), axis=0),
            ),
            tl.maximum(
                tl.max(tl.abs(values2), axis=0),
                tl.max(tl.abs(values3), axis=0),
            ),
        )
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)

        blob_base = (panel * groups + group) * 136
        tl.store(
            lhs_scale_ptr
            + blob_base // 2
            + panel_row
            + tl.arange(0, 1),
            scale.to(tl.float16),
        )
        data_base = lhs_data_ptr + blob_base + 8 + panel_row * 8
        tl.store(
            data_base + lanes,
            _quantize_symmetric_i8(values0, inv_scale),
        )
        tl.store(
            data_base + 32 + lanes,
            _quantize_symmetric_i8(values1, inv_scale),
        )
        tl.store(
            data_base + 64 + lanes,
            _quantize_symmetric_i8(values2, inv_scale),
        )
        tl.store(
            data_base + 96 + lanes,
            _quantize_symmetric_i8(values3, inv_scale),
        )


@triton.jit
def _q4_lhs_row_absmax(x_base, row_ok):
    """Reduce one BF16 K32 block while keeping only K8 slices live."""
    lanes = tl.arange(0, 8)
    values0 = tl.load(x_base + lanes).to(tl.float32)
    values1 = tl.load(x_base + 8 + lanes).to(tl.float32)
    values2 = tl.load(x_base + 16 + lanes).to(tl.float32)
    values3 = tl.load(x_base + 24 + lanes).to(tl.float32)
    values0 = tl.where(row_ok, values0, 0.0)
    values1 = tl.where(row_ok, values1, 0.0)
    values2 = tl.where(row_ok, values2, 0.0)
    values3 = tl.where(row_ok, values3, 0.0)
    return tl.maximum(
        tl.maximum(
            tl.max(tl.abs(values0), axis=0),
            tl.max(tl.abs(values1), axis=0),
        ),
        tl.maximum(
            tl.max(tl.abs(values2), axis=0),
            tl.max(tl.abs(values3), axis=0),
        ),
    )


@triton.jit
def _q4_lhs_full_row_absmax(x_base):
    """Reduce one known-valid BF16 K32 block without a tail predicate."""
    lanes = tl.arange(0, 8)
    values0 = tl.load(x_base + lanes).to(tl.float32)
    values1 = tl.load(x_base + 8 + lanes).to(tl.float32)
    values2 = tl.load(x_base + 16 + lanes).to(tl.float32)
    values3 = tl.load(x_base + 24 + lanes).to(tl.float32)
    return tl.maximum(
        tl.maximum(
            tl.max(tl.abs(values0), axis=0),
            tl.max(tl.abs(values1), axis=0),
        ),
        tl.maximum(
            tl.max(tl.abs(values2), axis=0),
            tl.max(tl.abs(values3), axis=0),
        ),
    )


@triton.jit
def _q4_lhs_full_row_absmax_bf16_bits(x_base):
    """Reduce finite BF16 magnitudes in their monotonic integer encoding."""
    lanes = tl.arange(0, 8)
    values0 = tl.load(x_base + lanes)
    values1 = tl.load(x_base + 8 + lanes)
    values2 = tl.load(x_base + 16 + lanes)
    values3 = tl.load(x_base + 24 + lanes)
    bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    max01 = tl.where(bits0 > bits1, bits0, bits1)
    max23 = tl.where(bits2 > bits3, bits2, bits3)
    lane_max = tl.where(max01 > max23, max01, max23)
    absmax_bits = tl.max(lane_max, axis=0)
    return (
        absmax_bits.to(tl.uint16)
        .to(tl.bfloat16, bitcast=True)
        .to(tl.float32)
    )


@triton.jit
def _q4_lhs_quantize_row_pair(
    x0_base,
    x1_base,
    data_base,
    row0_ok,
    row1_ok,
    inv0,
    inv1,
):
    """Quantize two rows and combine adjacent K8 stores into one K16."""
    lanes = tl.arange(0, 8)
    store_lanes = tl.arange(0, 16)
    values0 = tl.load(x0_base + lanes).to(tl.float32)
    values1 = tl.load(x1_base + lanes).to(tl.float32)
    values0 = tl.where(row0_ok, values0, 0.0)
    values1 = tl.where(row1_ok, values1, 0.0)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(data_base + store_lanes, quant01)

    values0 = tl.load(x0_base + 8 + lanes).to(tl.float32)
    values1 = tl.load(x1_base + 8 + lanes).to(tl.float32)
    values0 = tl.where(row0_ok, values0, 0.0)
    values1 = tl.where(row1_ok, values1, 0.0)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(
        data_base + 32 + store_lanes,
        quant01,
    )

    values0 = tl.load(x0_base + 16 + lanes).to(tl.float32)
    values1 = tl.load(x1_base + 16 + lanes).to(tl.float32)
    values0 = tl.where(row0_ok, values0, 0.0)
    values1 = tl.where(row1_ok, values1, 0.0)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(
        data_base + 64 + store_lanes,
        quant01,
    )

    values0 = tl.load(x0_base + 24 + lanes).to(tl.float32)
    values1 = tl.load(x1_base + 24 + lanes).to(tl.float32)
    values0 = tl.where(row0_ok, values0, 0.0)
    values1 = tl.where(row1_ok, values1, 0.0)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(
        data_base + 96 + store_lanes,
        quant01,
    )


@triton.jit
def _q4_lhs_quantize_full_row_pair(
    x0_base,
    x1_base,
    data_base,
    inv0,
    inv1,
):
    """Quantize two known-valid rows and emit adjacent K8 pairs as K16."""
    lanes = tl.arange(0, 8)
    store_lanes = tl.arange(0, 16)
    values0 = tl.load(x0_base + lanes).to(tl.float32)
    values1 = tl.load(x1_base + lanes).to(tl.float32)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(data_base + store_lanes, quant01)

    values0 = tl.load(x0_base + 8 + lanes).to(tl.float32)
    values1 = tl.load(x1_base + 8 + lanes).to(tl.float32)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(data_base + 32 + store_lanes, quant01)

    values0 = tl.load(x0_base + 16 + lanes).to(tl.float32)
    values1 = tl.load(x1_base + 16 + lanes).to(tl.float32)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(data_base + 64 + store_lanes, quant01)

    values0 = tl.load(x0_base + 24 + lanes).to(tl.float32)
    values1 = tl.load(x1_base + 24 + lanes).to(tl.float32)
    quant0 = _quantize_symmetric_i8(values0, inv0)
    quant1 = _quantize_symmetric_i8(values1, inv1)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    tl.store(data_base + 96 + store_lanes, quant01)


@triton.jit
def _q4_lhs_quantize_full_panel4_from_absmax(
    x0_base,
    x1_base,
    x2_base,
    x3_base,
    data_base,
    inv0,
    inv1,
    inv2,
    inv3,
):
    """Quantize one full M4 panel and expose contiguous K32 stores."""
    lanes = tl.arange(0, 8)
    store_lanes = tl.arange(0, 32)
    for offset in tl.static_range(0, 32, 8):
        values0 = tl.load(x0_base + offset + lanes).to(tl.float32)
        values1 = tl.load(x1_base + offset + lanes).to(tl.float32)
        values2 = tl.load(x2_base + offset + lanes).to(tl.float32)
        values3 = tl.load(x3_base + offset + lanes).to(tl.float32)
        quant0 = _quantize_symmetric_i8_from_absmax(values0, inv0)
        quant1 = _quantize_symmetric_i8_from_absmax(values1, inv1)
        quant2 = _quantize_symmetric_i8_from_absmax(values2, inv2)
        quant3 = _quantize_symmetric_i8_from_absmax(values3, inv3)
        quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
        quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
        quant0123 = (
            tl.join(quant01, quant23).permute(1, 0).reshape((32,))
        )
        tl.store(data_base + offset * 4 + store_lanes, quant0123)


@triton.jit
def _q4_lhs_quantize_full_row_from_absmax(
    x_base,
    data_base,
    inv_scale,
):
    """Quantize one valid K32 row into the compact decode layout."""
    lanes = tl.arange(0, 8)
    values0 = tl.load(x_base + lanes).to(tl.float32)
    values1 = tl.load(x_base + 8 + lanes).to(tl.float32)
    values2 = tl.load(x_base + 16 + lanes).to(tl.float32)
    values3 = tl.load(x_base + 24 + lanes).to(tl.float32)
    quant0 = _quantize_symmetric_i8_from_absmax(values0, inv_scale)
    quant1 = _quantize_symmetric_i8_from_absmax(values1, inv_scale)
    quant2 = _quantize_symmetric_i8_from_absmax(values2, inv_scale)
    quant3 = _quantize_symmetric_i8_from_absmax(values3, inv_scale)
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
    quantized = (
        tl.join(quant01, quant23).permute(1, 0).reshape((32,))
    )
    tl.store(data_base + tl.arange(0, 32), quantized)


@triton.jit
def _q4_rmsnorm_bf16_values8(x_base, weight_base, offset, rrms):
    """Materialize the exact Qwen BF16 RMSNorm value for one K8 slice."""
    lanes = offset + tl.arange(0, 8)
    values = tl.load(x_base + lanes).to(tl.float32)
    weight = tl.load(weight_base + lanes).to(tl.float32)
    # Qwen first rounds x*rrms to the input dtype, then performs the BF16
    # weight multiply.  Make both boundaries explicit so activation packing
    # observes the same BF16 bits as the unfused graph.
    normalized = (values * rrms).to(tl.bfloat16).to(tl.float32)
    return (normalized * weight).to(tl.bfloat16)


@triton.jit
def _q4_rmsnorm_stored_bf16_values8(
    summed_base, weight_base, offset, rrms
):
    """Normalize a materialized BF16 residual sum for one K8 slice."""
    lanes = offset + tl.arange(0, 8)
    values = tl.load(summed_base + lanes).to(tl.float32)
    weight = tl.load(weight_base + lanes).to(tl.float32)
    normalized = (values * rrms).to(tl.bfloat16).to(tl.float32)
    return (normalized * weight).to(tl.bfloat16)


@triton.jit
def _q4_bf16_values_absmax(values0, values1, values2, values3):
    """Reduce four finite BF16 K8 values through their magnitude bits."""
    bits0 = (values0.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits1 = (values1.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits2 = (values2.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    bits3 = (values3.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.int16)
    max01 = tl.where(bits0 > bits1, bits0, bits1)
    max23 = tl.where(bits2 > bits3, bits2, bits3)
    lane_max = tl.where(max01 > max23, max01, max23)
    absmax_bits = tl.max(lane_max, axis=0)
    return (
        absmax_bits.to(tl.uint16)
        .to(tl.bfloat16, bitcast=True)
        .to(tl.float32)
    )


@triton.jit
def _q4_store_quantized_bf16_values(
    data_base, values0, values1, values2, values3, inv_scale
):
    """Quantize four already-rounded BF16 K8 values into decode K32."""
    quant0 = _quantize_symmetric_i8_from_absmax(
        values0.to(tl.float32), inv_scale
    )
    quant1 = _quantize_symmetric_i8_from_absmax(
        values1.to(tl.float32), inv_scale
    )
    quant2 = _quantize_symmetric_i8_from_absmax(
        values2.to(tl.float32), inv_scale
    )
    quant3 = _quantize_symmetric_i8_from_absmax(
        values3.to(tl.float32), inv_scale
    )
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
    quantized = tl.join(quant01, quant23).permute(1, 0).reshape((32,))
    tl.store(data_base + tl.arange(0, 32), quantized)


@triton.jit
def _pack_lhs_qsi8d32p_panel4_scalar_kernel(
    x_ptr,
    lhs_scale_ptr,
    lhs_data_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
    FULL_PANEL: tl.constexpr = False,
):
    """Vectorize M4 scales; specialize away predicates for full panels."""
    panel = tl.program_id(0)
    row0 = panel * 4
    groups: tl.constexpr = K // 32

    for group in tl.range(0, groups, loop_unroll_factor=1):
        if FULL_PANEL:
            x0 = x_ptr + row0 * stride_xm + group * 32
            x1 = x0 + stride_xm
            x2 = x1 + stride_xm
            x3 = x2 + stride_xm
        else:
            source0 = tl.minimum(row0, M - 1)
            source1 = tl.minimum(row0 + 1, M - 1)
            source2 = tl.minimum(row0 + 2, M - 1)
            source3 = tl.minimum(row0 + 3, M - 1)
            x0 = x_ptr + source0 * stride_xm + group * 32
            x1 = x_ptr + source1 * stride_xm + group * 32
            x2 = x_ptr + source2 * stride_xm + group * 32
            x3 = x_ptr + source3 * stride_xm + group * 32
        if FULL_PANEL:
            max0 = _q4_lhs_full_row_absmax_bf16_bits(x0)
            max1 = _q4_lhs_full_row_absmax_bf16_bits(x1)
            max2 = _q4_lhs_full_row_absmax_bf16_bits(x2)
            max3 = _q4_lhs_full_row_absmax_bf16_bits(x3)
        else:
            max0 = _q4_lhs_row_absmax(x0, row0 < M)
            max1 = _q4_lhs_row_absmax(x1, row0 + 1 < M)
            max2 = _q4_lhs_row_absmax(x2, row0 + 2 < M)
            max3 = _q4_lhs_row_absmax(x3, row0 + 3 < M)
        absmax = tl.join(
            tl.join(max0, max2), tl.join(max1, max3)
        ).reshape((4,))
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = (panel * groups + group) * 136
        tl.store(
            lhs_scale_ptr + blob_base // 2 + tl.arange(0, 4),
            scale.to(tl.float16),
        )
        inv_even, inv_odd = tl.split(inv_scale.reshape((2, 2)))
        inv0, inv2 = tl.split(inv_even)
        inv1, inv3 = tl.split(inv_odd)
        data_base = lhs_data_ptr + blob_base + 8
        if FULL_PANEL:
            _q4_lhs_quantize_full_panel4_from_absmax(
                x0,
                x1,
                x2,
                x3,
                data_base,
                inv0,
                inv1,
                inv2,
                inv3,
            )
        else:
            _q4_lhs_quantize_row_pair(
                x0, x1, data_base, row0 < M, row0 + 1 < M, inv0, inv1
            )
            _q4_lhs_quantize_row_pair(
                x2,
                x3,
                data_base + 16,
                row0 + 2 < M,
                row0 + 3 < M,
                inv2,
                inv3,
            )


@triton.jit
def _pack_lhs_qsi8d32p_decode_kernel(
    x_ptr,
    lhs_scale_ptr,
    lhs_data_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Pack exact-grid rows into the compact 34-byte decode ABI."""
    row = tl.program_id(0)
    groups: tl.constexpr = K // 32
    for group in tl.range(0, groups, loop_unroll_factor=1):
        x_base = x_ptr + row * stride_xm + group * 32
        absmax = _q4_lhs_full_row_absmax_bf16_bits(x_base)
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = (row * groups + group) * 34
        tl.store(
            lhs_scale_ptr + blob_base // 2,
            scale.to(tl.float16),
        )
        _q4_lhs_quantize_full_row_from_absmax(
            x_base,
            lhs_data_ptr + blob_base + 2,
            inv_scale,
        )


@triton.jit
def _pack_lhs_qsi8d32p_asym_decode_kernel(
    x_ptr,
    lhs_packed_ptr,
    stride_xm,
    K: tl.constexpr,
    COMPACT: tl.constexpr = False,
):
    """Pack token-wise asymmetric A8 with shared or per-K32 metadata."""
    row = tl.program_id(0)
    groups: tl.constexpr = K // 32
    group_stride: tl.constexpr = 36
    lhs_packed_ptr = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    source_row = x_ptr + row * stride_xm
    scale, inv_scale, zero_point = _q4_token_asymmetric_qparams_bf16(
        source_row, K=K
    )
    if COMPACT:
        packed_row = lhs_packed_ptr + row * (4 + K)
        tl.store(
            packed_row.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 1),
            scale,
        )
        tl.store(packed_row + 2 + tl.arange(0, 1), zero_point)
    else:
        packed_row = lhs_packed_ptr + row * groups * group_stride
    for group in tl.range(0, groups, loop_unroll_factor=1):
        if COMPACT:
            data = packed_row + 4 + group * 32
        else:
            blob = packed_row + group * group_stride
            tl.store(
                blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 1),
                scale,
            )
            tl.store(blob + 2 + tl.arange(0, 1), zero_point)
            data = blob + 4
        _q4_store_token_asymmetric_k32(
            data,
            source_row + group * 32,
            inv_scale,
            zero_point,
        )


@triton.jit(do_not_specialize=["rms_eps"])
def _pack_lhs_qsi8d32p_asym_rmsnorm_decode_kernel(
    x_ptr,
    rms_weight_ptr,
    lhs_packed_ptr,
    stride_xm,
    rms_eps,
    K: tl.constexpr,
    NORM_TILE: tl.constexpr,
    COMPACT: tl.constexpr = False,
):
    """Fuse BF16 RMSNorm and faithful token-asymmetric A8 packing."""
    row = tl.program_id(0)
    source_row = x_ptr + row * stride_xm
    lhs_packed_ptr = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    groups: tl.constexpr = K // 32
    group_stride: tl.constexpr = 36
    lanes = tl.arange(0, NORM_TILE)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for start in tl.range(0, K, NORM_TILE, loop_unroll_factor=1):
        values = tl.load(source_row + start + lanes).to(tl.float32)
        sum_sq += tl.sum(values * values, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / K + rms_eps)

    row_min = tl.full((1,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((1,), -3.4028234663852886e38, tl.float32)
    for group in tl.range(0, groups, loop_unroll_factor=1):
        x_base = source_row + group * 32
        weight_base = rms_weight_ptr + group * 32
        values0 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 0, rrms)
        values1 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 8, rrms)
        values2 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 16, rrms)
        values3 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 24, rrms)
        group_min, group_max = _q4_bf16_values_minmax(
            values0, values1, values2, values3
        )
        row_min = tl.minimum(row_min, group_min)
        row_max = tl.maximum(row_max, group_max)
    scale, inv_scale, zero_point = _q4_asymmetric_qparams_from_minmax(
        row_min, row_max
    )

    if COMPACT:
        packed_row = lhs_packed_ptr + row * (4 + K)
        tl.store(
            packed_row.to(tl.pointer_type(tl.bfloat16))
            + tl.arange(0, 1),
            scale,
        )
        tl.store(packed_row + 2 + tl.arange(0, 1), zero_point)
    else:
        packed_row = lhs_packed_ptr + row * groups * group_stride
    for group in tl.range(0, groups, loop_unroll_factor=1):
        x_base = source_row + group * 32
        weight_base = rms_weight_ptr + group * 32
        values0 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 0, rrms)
        values1 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 8, rrms)
        values2 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 16, rrms)
        values3 = _q4_rmsnorm_bf16_values8(x_base, weight_base, 24, rrms)
        if COMPACT:
            data_ptr = packed_row + 4 + group * 32
        else:
            blob = packed_row + group * group_stride
            tl.store(
                blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 1), scale
            )
            tl.store(blob + 2 + tl.arange(0, 1), zero_point)
            data_ptr = blob + 4
        _q4_store_token_asymmetric_bf16_values(
            data_ptr,
            values0,
            values1,
            values2,
            values3,
            inv_scale,
            zero_point,
        )


@triton.jit(do_not_specialize=["rms_eps"])
def _pack_lhs_qsi8d32p_asym_add_rmsnorm_decode_kernel(
    x_ptr,
    residual_ptr,
    rms_weight_ptr,
    lhs_packed_ptr,
    updated_residual_ptr,
    stride_xm,
    rms_eps,
    K: tl.constexpr,
    NORM_TILE: tl.constexpr,
    COMPACT: tl.constexpr = False,
):
    """Fuse residual add, RMSNorm and token-asymmetric A8 packing."""
    row = tl.program_id(0)
    input_row = x_ptr + row * stride_xm
    residual_row = residual_ptr + row * stride_xm
    summed_row = updated_residual_ptr + row * stride_xm
    lhs_packed_ptr = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    groups: tl.constexpr = K // 32
    group_stride: tl.constexpr = 36
    lanes = tl.arange(0, NORM_TILE)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for start in tl.range(0, K, NORM_TILE, loop_unroll_factor=1):
        summed = (
            tl.load(input_row + start + lanes).to(tl.float32)
            + tl.load(residual_row + start + lanes).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(summed_row + start + lanes, summed)
        values = summed.to(tl.float32)
        sum_sq += tl.sum(values * values, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / K + rms_eps)

    row_min = tl.full((1,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((1,), -3.4028234663852886e38, tl.float32)
    for group in tl.range(0, groups, loop_unroll_factor=1):
        summed_base = summed_row + group * 32
        weight_base = rms_weight_ptr + group * 32
        values0 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 0, rrms
        )
        values1 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 8, rrms
        )
        values2 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 16, rrms
        )
        values3 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 24, rrms
        )
        group_min, group_max = _q4_bf16_values_minmax(
            values0, values1, values2, values3
        )
        row_min = tl.minimum(row_min, group_min)
        row_max = tl.maximum(row_max, group_max)
    scale, inv_scale, zero_point = _q4_asymmetric_qparams_from_minmax(
        row_min, row_max
    )

    if COMPACT:
        packed_row = lhs_packed_ptr + row * (4 + K)
        tl.store(
            packed_row.to(tl.pointer_type(tl.bfloat16))
            + tl.arange(0, 1),
            scale,
        )
        tl.store(packed_row + 2 + tl.arange(0, 1), zero_point)
    else:
        packed_row = lhs_packed_ptr + row * groups * group_stride
    for group in tl.range(0, groups, loop_unroll_factor=1):
        summed_base = summed_row + group * 32
        weight_base = rms_weight_ptr + group * 32
        values0 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 0, rrms
        )
        values1 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 8, rrms
        )
        values2 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 16, rrms
        )
        values3 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 24, rrms
        )
        if COMPACT:
            data_ptr = packed_row + 4 + group * 32
        else:
            blob = packed_row + group * group_stride
            tl.store(
                blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 1), scale
            )
            tl.store(blob + 2 + tl.arange(0, 1), zero_point)
            data_ptr = blob + 4
        _q4_store_token_asymmetric_bf16_values(
            data_ptr,
            values0,
            values1,
            values2,
            values3,
            inv_scale,
            zero_point,
        )


@triton.jit
def _q4_decode_sdot_kai_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    LHS_PARTITIONED: tl.constexpr = False,
):
    """One-row KAI-layout Q4 GEMV lowered to spill-free SDOT/ADDP."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    tile_count = range_end - range_begin
    tiles_per_partition = (tile_count + partitions - 1) // partitions
    local_begin = range_begin + partition * tiles_per_partition
    local_end = tl.minimum(range_end, local_begin + tiles_per_partition)
    groups: tl.constexpr = K // 32
    lhs_group_stride: tl.constexpr = 34
    rhs_group_stride: tl.constexpr = 72
    rhs_tile_stride: tl.constexpr = groups * rhs_group_stride
    q_lanes = tl.arange(0, 16)
    x_lanes = tl.arange(0, 8)
    output_lanes = tl.arange(0, 4)

    for tile in range(local_begin, local_end):
        result = tl.zeros((4,), dtype=tl.float32)
        if LHS_PARTITIONED:
            lhs_group_ptr = lhs_packed_ptr + (
                row * partitions + partition
            ) * groups * lhs_group_stride
        else:
            lhs_group_ptr = (
                lhs_packed_ptr + row * groups * lhs_group_stride
            )
        rhs_group_ptr = rhs_packed_ptr + tile * rhs_tile_stride
        for group in tl.range(0, groups, loop_unroll_factor=UNROLL):
            lhs_scale = tl.load(
                lhs_group_ptr.to(tl.pointer_type(tl.float16))
            ).to(tl.float32)
            rhs_scale = tl.load(
                rhs_group_ptr.to(tl.pointer_type(tl.float16))
                + output_lanes
            ).to(tl.float32)

            q0 = tl.load(rhs_group_ptr + 8 + q_lanes)
            q1 = tl.load(rhs_group_ptr + 24 + q_lanes)
            q2 = tl.load(rhs_group_ptr + 40 + q_lanes)
            q3 = tl.load(rhs_group_ptr + 56 + q_lanes)
            q0_low = (q0 << 4).to(tl.int8).reshape((4, 4))
            q1_low = (q1 << 4).to(tl.int8).reshape((4, 4))
            q2_low = (q2 << 4).to(tl.int8).reshape((4, 4))
            q3_low = (q3 << 4).to(tl.int8).reshape((4, 4))
            q0_high = (q0 & 0xF0).to(tl.int8).reshape((4, 4))
            q1_high = (q1 & 0xF0).to(tl.int8).reshape((4, 4))
            q2_high = (q2 & 0xF0).to(tl.int8).reshape((4, 4))
            q3_high = (q3 & 0xF0).to(tl.int8).reshape((4, 4))

            x_ptr = lhs_group_ptr + 2
            x0 = tl.load(x_ptr + x_lanes).to(tl.int8).reshape((2, 4))
            x1 = tl.load(x_ptr + 8 + x_lanes).to(tl.int8).reshape((2, 4))
            x2 = tl.load(x_ptr + 16 + x_lanes).to(tl.int8).reshape((2, 4))
            x3 = tl.load(x_ptr + 24 + x_lanes).to(tl.int8).reshape((2, 4))
            x0 = tl.join(x0.reshape((8,)), x0.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x1 = tl.join(x1.reshape((8,)), x1.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x2 = tl.join(x2.reshape((8,)), x2.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x3 = tl.join(x3.reshape((8,)), x3.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))

            partial01 = tl.sum(
                q0_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial23 = tl.sum(
                q1_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q0_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q1_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )
            partial = tl.join(partial01, partial23).permute(
                1, 0
            ).reshape((4, 2))
            dot_scaled16 = tl.sum(partial, axis=1)
            result += (
                dot_scaled16.to(tl.float32) * (1.0 / 16.0)
                * lhs_scale * rhs_scale
            )
            lhs_group_ptr += lhs_group_stride
            rhs_group_ptr += rhs_group_stride

        tl.store(
            out_ptr + row * N + tile * 4 + output_lanes,
            result.to(tl.bfloat16),
        )


@triton.jit
def _q4_decode_asym_sdot_kai_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    LHS_PARTITIONED: tl.constexpr = False,
    LHS_COMPACT: tl.constexpr = False,
    LHS_KAI: tl.constexpr = False,
):
    """Token-asymmetric A8 x signed Q4 GEMV with visible zp correction."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    tile_count = range_end - range_begin
    tiles_per_partition = (tile_count + partitions - 1) // partitions
    local_begin = range_begin + partition * tiles_per_partition
    local_end = tl.minimum(range_end, local_begin + tiles_per_partition)
    groups: tl.constexpr = K // 32
    lhs_group_stride: tl.constexpr = 36
    rhs_group_stride: tl.constexpr = 80
    rhs_tile_stride: tl.constexpr = groups * rhs_group_stride
    q_lanes = tl.arange(0, 16)
    x_lanes = tl.arange(0, 8)
    output_lanes = tl.arange(0, 4)
    lhs_packed_ptr = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))

    if LHS_COMPACT:
        lhs_row_stride: tl.constexpr = K + (8 if LHS_KAI else 4)
        if LHS_PARTITIONED:
            lhs_row_ptr = lhs_packed_ptr + (
                row * partitions + partition
            ) * lhs_row_stride
        else:
            lhs_row_ptr = lhs_packed_ptr + row * lhs_row_stride
        if LHS_KAI:
            shared_lhs_scale = tl.load(
                lhs_row_ptr.to(tl.pointer_type(tl.float32))
            )
            shared_zero_point = tl.load(lhs_row_ptr + 4).to(tl.int8).to(
                tl.int32
            )
            lhs_data_ptr = lhs_row_ptr + 8
        else:
            shared_lhs_scale = tl.load(
                lhs_row_ptr.to(tl.pointer_type(tl.bfloat16))
            ).to(tl.float32)
            shared_zero_point = tl.load(lhs_row_ptr + 2).to(tl.int8).to(
                tl.int32
            )
            lhs_data_ptr = lhs_row_ptr + 4

    for tile in range(local_begin, local_end):
        result = tl.zeros((4,), dtype=tl.float32)
        if LHS_COMPACT:
            lhs_group_ptr = lhs_data_ptr
        else:
            if LHS_PARTITIONED:
                lhs_group_ptr = lhs_packed_ptr + (
                    row * partitions + partition
                ) * groups * lhs_group_stride
            else:
                lhs_group_ptr = (
                    lhs_packed_ptr + row * groups * lhs_group_stride
                )
        rhs_group_ptr = rhs_packed_ptr + tile * rhs_tile_stride
        for group in tl.range(0, groups, loop_unroll_factor=UNROLL):
            if LHS_COMPACT:
                lhs_scale = shared_lhs_scale
                zero_point = shared_zero_point
                x_ptr = lhs_group_ptr
            else:
                lhs_scale = tl.load(
                    lhs_group_ptr.to(tl.pointer_type(tl.bfloat16))
                ).to(tl.float32)
                zero_point = tl.load(lhs_group_ptr + 2).to(tl.int8).to(
                    tl.int32
                )
                x_ptr = lhs_group_ptr + 4
            rhs_scale = tl.load(
                rhs_group_ptr.to(tl.pointer_type(tl.bfloat16)) + output_lanes
            ).to(tl.float32)
            rhs_sum_scaled16 = tl.load(
                (rhs_group_ptr + 72).to(tl.pointer_type(tl.int16))
                + output_lanes
            ).to(tl.int32)

            q0 = tl.load(rhs_group_ptr + 8 + q_lanes)
            q1 = tl.load(rhs_group_ptr + 24 + q_lanes)
            q2 = tl.load(rhs_group_ptr + 40 + q_lanes)
            q3 = tl.load(rhs_group_ptr + 56 + q_lanes)
            q0_low = (q0 << 4).to(tl.int8).reshape((4, 4))
            q1_low = (q1 << 4).to(tl.int8).reshape((4, 4))
            q2_low = (q2 << 4).to(tl.int8).reshape((4, 4))
            q3_low = (q3 << 4).to(tl.int8).reshape((4, 4))
            q0_high = (q0 & 0xF0).to(tl.int8).reshape((4, 4))
            q1_high = (q1 & 0xF0).to(tl.int8).reshape((4, 4))
            q2_high = (q2 & 0xF0).to(tl.int8).reshape((4, 4))
            q3_high = (q3 & 0xF0).to(tl.int8).reshape((4, 4))

            x0 = tl.load(x_ptr + x_lanes).to(tl.int8).reshape((2, 4))
            x1 = tl.load(x_ptr + 8 + x_lanes).to(tl.int8).reshape((2, 4))
            x2 = tl.load(x_ptr + 16 + x_lanes).to(tl.int8).reshape((2, 4))
            x3 = tl.load(x_ptr + 24 + x_lanes).to(tl.int8).reshape((2, 4))
            x0 = tl.join(x0.reshape((8,)), x0.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x1 = tl.join(x1.reshape((8,)), x1.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x2 = tl.join(x2.reshape((8,)), x2.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))
            x3 = tl.join(x3.reshape((8,)), x3.reshape((8,))).permute(
                1, 0
            ).reshape((4, 4))

            partial01 = tl.sum(
                q0_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial23 = tl.sum(
                q1_low.to(tl.int32) * x0.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_low.to(tl.int32) * x1.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q0_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q1_high.to(tl.int32) * x2.to(tl.int32), axis=1
            )
            partial01 += tl.sum(
                q2_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )
            partial23 += tl.sum(
                q3_high.to(tl.int32) * x3.to(tl.int32), axis=1
            )
            partial = tl.join(partial01, partial23).permute(
                1, 0
            ).reshape((4, 2))
            dot_scaled16 = tl.sum(partial, axis=1)
            corrected = dot_scaled16 - zero_point * rhs_sum_scaled16
            result += (
                corrected.to(tl.float32) * (1.0 / 16.0)
                * lhs_scale * rhs_scale
            )
            lhs_group_ptr += 32 if LHS_COMPACT else lhs_group_stride
            rhs_group_ptr += rhs_group_stride

        tl.store(
            out_ptr + row * N + tile * 4 + output_lanes,
            result.to(tl.bfloat16),
        )


@triton.jit
def _q4_decode_asym_g128_k32_lhs(lhs_data_ptr):
    """Load/rearrange one K32 activation once for one or more N4 tiles."""
    x_lanes = tl.arange(0, 8)
    x0 = tl.load(lhs_data_ptr + x_lanes).to(tl.int8).reshape((2, 4))
    x1 = tl.load(lhs_data_ptr + 8 + x_lanes).to(tl.int8).reshape((2, 4))
    x2 = tl.load(lhs_data_ptr + 16 + x_lanes).to(tl.int8).reshape((2, 4))
    x3 = tl.load(lhs_data_ptr + 24 + x_lanes).to(tl.int8).reshape((2, 4))
    x0 = tl.join(x0.reshape((8,)), x0.reshape((8,))).permute(
        1, 0
    ).reshape((4, 4))
    x1 = tl.join(x1.reshape((8,)), x1.reshape((8,))).permute(
        1, 0
    ).reshape((4, 4))
    x2 = tl.join(x2.reshape((8,)), x2.reshape((8,))).permute(
        1, 0
    ).reshape((4, 4))
    x3 = tl.join(x3.reshape((8,)), x3.reshape((8,))).permute(
        1, 0
    ).reshape((4, 4))
    return x0, x1, x2, x3


@triton.jit
def _q4_decode_asym_g128_k32_rhs_dot(rhs_data_ptr, x0, x1, x2, x3):
    """One compiler-visible packed-Q4 SDOT body with a reusable LHS."""
    q_lanes = tl.arange(0, 16)
    q0 = tl.load(rhs_data_ptr + q_lanes)
    q1 = tl.load(rhs_data_ptr + 16 + q_lanes)
    q2 = tl.load(rhs_data_ptr + 32 + q_lanes)
    q3 = tl.load(rhs_data_ptr + 48 + q_lanes)
    q0_low = (q0 << 4).to(tl.int8).reshape((4, 4))
    q1_low = (q1 << 4).to(tl.int8).reshape((4, 4))
    q2_low = (q2 << 4).to(tl.int8).reshape((4, 4))
    q3_low = (q3 << 4).to(tl.int8).reshape((4, 4))
    q0_high = (q0 & 0xF0).to(tl.int8).reshape((4, 4))
    q1_high = (q1 & 0xF0).to(tl.int8).reshape((4, 4))
    q2_high = (q2 & 0xF0).to(tl.int8).reshape((4, 4))
    q3_high = (q3 & 0xF0).to(tl.int8).reshape((4, 4))

    partial01 = tl.sum(q0_low.to(tl.int32) * x0.to(tl.int32), axis=1)
    partial23 = tl.sum(q1_low.to(tl.int32) * x0.to(tl.int32), axis=1)
    partial01 += tl.sum(q2_low.to(tl.int32) * x1.to(tl.int32), axis=1)
    partial23 += tl.sum(q3_low.to(tl.int32) * x1.to(tl.int32), axis=1)
    partial01 += tl.sum(q0_high.to(tl.int32) * x2.to(tl.int32), axis=1)
    partial23 += tl.sum(q1_high.to(tl.int32) * x2.to(tl.int32), axis=1)
    partial01 += tl.sum(q2_high.to(tl.int32) * x3.to(tl.int32), axis=1)
    partial23 += tl.sum(q3_high.to(tl.int32) * x3.to(tl.int32), axis=1)
    partial = tl.join(partial01, partial23).permute(1, 0).reshape((4, 2))
    return tl.sum(partial, axis=1)


@triton.jit
def _q4_decode_asym_g128_k32_dot(lhs_data_ptr, rhs_data_ptr):
    x0, x1, x2, x3 = _q4_decode_asym_g128_k32_lhs(lhs_data_ptr)
    return _q4_decode_asym_g128_k32_rhs_dot(
        rhs_data_ptr, x0, x1, x2, x3
    )


@triton.jit
def _q4_decode_asym_g128_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    residual_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    LHS_PARTITIONED: tl.constexpr = False,
    LHS_COMPACT: tl.constexpr = False,
    LHS_KAI: tl.constexpr = False,
    ADD_RESIDUAL: tl.constexpr = False,
    UNROLL: tl.constexpr = 1,
):
    """G128 Q4 decode: one scale/correction and four SDOT K32 bodies."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    tile_count = range_end - range_begin
    tiles_per_partition = (tile_count + partitions - 1) // partitions
    local_begin = range_begin + partition * tiles_per_partition
    local_end = tl.minimum(range_end, local_begin + tiles_per_partition)
    groups32: tl.constexpr = K // 32
    groups128: tl.constexpr = K // 128
    lhs_group_stride: tl.constexpr = 36
    rhs_group_stride: tl.constexpr = 264
    rhs_tile_stride: tl.constexpr = groups128 * rhs_group_stride + 16
    output_lanes = tl.arange(0, 4)
    lhs_packed_ptr = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))

    if LHS_PARTITIONED:
        if LHS_COMPACT:
            if LHS_KAI:
                lhs_row_ptr = lhs_packed_ptr + (
                    row * partitions + partition
                ) * (8 + K)
            else:
                lhs_row_ptr = lhs_packed_ptr + (
                    row * partitions + partition
                ) * (4 + K)
        else:
            lhs_row_ptr = lhs_packed_ptr + (
                row * partitions + partition
            ) * groups32 * lhs_group_stride
    else:
        if LHS_COMPACT:
            if LHS_KAI:
                lhs_row_ptr = lhs_packed_ptr + row * (8 + K)
            else:
                lhs_row_ptr = lhs_packed_ptr + row * (4 + K)
        else:
            lhs_row_ptr = lhs_packed_ptr + row * groups32 * lhs_group_stride
    if LHS_KAI:
        lhs_scale = tl.load(
            lhs_row_ptr.to(tl.pointer_type(tl.float32))
        )
        zero_point = tl.load(lhs_row_ptr + 4).to(tl.int8).to(tl.int32)
    else:
        lhs_scale = tl.load(
            lhs_row_ptr.to(tl.pointer_type(tl.bfloat16))
        ).to(tl.float32)
        zero_point = tl.load(lhs_row_ptr + 2).to(tl.int8).to(tl.int32)

    for tile in range(local_begin, local_end):
        result = tl.zeros((4,), dtype=tl.float32)
        rhs_tile_ptr = rhs_packed_ptr + tile * rhs_tile_stride
        rhs_group_ptr = rhs_tile_ptr
        if LHS_COMPACT:
            if LHS_KAI:
                lhs_group_ptr = lhs_row_ptr + 8
            else:
                lhs_group_ptr = lhs_row_ptr + 4
        else:
            lhs_group_ptr = lhs_row_ptr
        for group in tl.range(0, groups128, loop_unroll_factor=UNROLL):
            rhs_scale = tl.load(
                rhs_group_ptr.to(tl.pointer_type(tl.bfloat16)) + output_lanes
            ).to(tl.float32)
            dot_scaled16 = tl.zeros((4,), dtype=tl.int32)
            for subgroup in tl.range(0, 4, loop_unroll_factor=1):
                if LHS_COMPACT:
                    lhs_data_ptr = lhs_group_ptr + subgroup * 32
                else:
                    lhs_data_ptr = (
                        lhs_group_ptr + subgroup * lhs_group_stride + 4
                    )
                dot_scaled16 += _q4_decode_asym_g128_k32_dot(
                    lhs_data_ptr,
                    rhs_group_ptr + 8 + subgroup * 64,
                )
            result += dot_scaled16.to(tl.float32) * rhs_scale
            if LHS_COMPACT:
                lhs_group_ptr += 128
            else:
                lhs_group_ptr += 4 * lhs_group_stride
            rhs_group_ptr += rhs_group_stride

        weighted_sum_scaled16 = tl.load(
            (rhs_tile_ptr + groups128 * rhs_group_stride).to(
                tl.pointer_type(tl.float32)
            ) + output_lanes
        )
        result -= zero_point.to(tl.float32) * weighted_sum_scaled16
        result *= lhs_scale * (1.0 / 16.0)

        output_offsets = row * N + tile * 4 + output_lanes
        output = result.to(tl.bfloat16)
        if ADD_RESIDUAL:
            residual = tl.load(residual_ptr + output_offsets)
            output = (
                output.to(tl.float32) + residual.to(tl.float32)
            ).to(tl.bfloat16)
        tl.store(out_ptr + output_offsets, output)


@triton.jit
def _q4_fused_decode_asym_g128_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
):
    """Single-entry token-asymmetric pack plus G128 Q4 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups32: tl.constexpr = K // 32
    lhs_row_stride: tl.constexpr = 4 + K
    source_row = x_ptr + row * stride_xm
    scale, inv_scale, zero_point = _q4_token_asymmetric_qparams_bf16(
        source_row, K=K
    )
    scratch_base = (
        (row * partitions + partition) * lhs_row_stride
    )
    scratch_row = workspace_bytes + scratch_base
    tl.store(
        scratch_row.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 1), scale
    )
    tl.store(scratch_row + 2 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32(
            scratch_row + 4 + group * 32,
            source_row + group * 32,
            inv_scale,
            zero_point,
        )
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_decode_asym_g128_sdot_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        LHS_PARTITIONED=True,
        LHS_COMPACT=True,
        ADD_RESIDUAL=False,
        UNROLL=UNROLL,
    )


@triton.jit
def _q4_fused_decode_asym_g128_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
):
    """KleidiAI-compatible FP32 activation pack plus Triton G128 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups32: tl.constexpr = K // 32
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_base = (row * partitions + partition) * lhs_row_stride
    scratch_row = workspace_bytes + scratch_base
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_decode_asym_g128_sdot_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        LHS_PARTITIONED=True,
        LHS_COMPACT=True,
        LHS_KAI=True,
        ADD_RESIDUAL=False,
        UNROLL=UNROLL,
    )


@triton.jit
def _q4_fused_decode_asym_g128_pair_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs0_packed_ptr,
    rhs1_packed_ptr,
    output_byte_offset,
    stride_xm,
    K: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    UNROLL0: tl.constexpr = 1,
    UNROLL1: tl.constexpr = 1,
):
    """Pack one activation and evaluate the two GDN G128 projections."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_row = workspace_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1), scale
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, K // 32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )

    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_decode_asym_g128_sdot_kernel(
        workspace_bytes,
        rhs0_packed_ptr,
        out_ptr,
        out_ptr,
        0,
        N0 // 4,
        K=K,
        N=N0,
        LHS_PARTITIONED=True,
        LHS_COMPACT=True,
        LHS_KAI=True,
        ADD_RESIDUAL=False,
        UNROLL=UNROLL0,
    )
    _q4_decode_asym_g128_sdot_kernel(
        workspace_bytes,
        rhs1_packed_ptr,
        out_ptr + N0,
        out_ptr + N0,
        0,
        N1 // 4,
        K=K,
        N=N1,
        LHS_PARTITIONED=True,
        LHS_COMPACT=True,
        LHS_KAI=True,
        ADD_RESIDUAL=False,
        UNROLL=UNROLL1,
    )


@triton.jit
def _q4_decode_asym_g128_kai_shared_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
):
    """G128 decode wrapper for a shared KAI A8 row."""
    _q4_decode_asym_g128_sdot_kernel(
        lhs_packed_ptr,
        rhs_packed_ptr,
        out_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        LHS_COMPACT=True,
        LHS_KAI=True,
        ADD_RESIDUAL=False,
        UNROLL=UNROLL,
    )


@triton.jit
def _q4_decode_asym_g128_stealing_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    counter_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    STEAL_CHUNK: tl.constexpr = 32,
    LHS_PARTITIONED: tl.constexpr = True,
):
    """Evaluate G128 GEMV with a shared N4 work counter."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    groups128: tl.constexpr = K // 128
    lhs_row_stride: tl.constexpr = 8 + K
    lhs_bytes = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    if LHS_PARTITIONED:
        scratch_row = lhs_bytes + (
            row * partitions + partition
        ) * lhs_row_stride
    else:
        scratch_row = lhs_bytes + row * lhs_row_stride
    rhs_group_stride: tl.constexpr = 264
    rhs_tile_stride: tl.constexpr = groups128 * rhs_group_stride + 16
    output_lanes = tl.arange(0, 4)
    lhs_scale = tl.load(scratch_row.to(tl.pointer_type(tl.float32)))
    lhs_zero_point = tl.load(scratch_row + 4).to(tl.int8).to(tl.int32)
    lhs_data_ptr = scratch_row + 8

    local_begin = range_begin + tl.atomic_add(counter_ptr, STEAL_CHUNK)
    while local_begin < range_end:
        local_end = tl.minimum(range_end, local_begin + STEAL_CHUNK)
        for tile in range(local_begin, local_end):
            result = tl.zeros((4,), dtype=tl.float32)
            rhs_tile_ptr = rhs_packed_ptr + tile * rhs_tile_stride
            rhs_group_ptr = rhs_tile_ptr
            lhs_group_ptr = lhs_data_ptr
            for group in tl.range(
                0, groups128, loop_unroll_factor=UNROLL
            ):
                rhs_scale = tl.load(
                    rhs_group_ptr.to(tl.pointer_type(tl.bfloat16))
                    + output_lanes
                ).to(tl.float32)
                dot_scaled16 = tl.zeros((4,), dtype=tl.int32)
                for subgroup in tl.range(0, 4, loop_unroll_factor=1):
                    dot_scaled16 += _q4_decode_asym_g128_k32_dot(
                        lhs_group_ptr + subgroup * 32,
                        rhs_group_ptr + 8 + subgroup * 64,
                    )
                result += dot_scaled16.to(tl.float32) * rhs_scale
                lhs_group_ptr += 128
                rhs_group_ptr += rhs_group_stride

            weighted_sum_scaled16 = tl.load(
                (rhs_tile_ptr + groups128 * rhs_group_stride).to(
                    tl.pointer_type(tl.float32)
                )
                + output_lanes
            )
            result -= (
                lhs_zero_point.to(tl.float32) * weighted_sum_scaled16
            )
            result *= lhs_scale * (1.0 / 16.0)
            tl.store(
                out_ptr + row * N + tile * 4 + output_lanes,
                result.to(tl.bfloat16),
            )
        local_begin = range_begin + tl.atomic_add(counter_ptr, STEAL_CHUNK)


@triton.jit
def _q4_decode_asym_g128_kai_shared_stealing_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    counter_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    STEAL_CHUNK: tl.constexpr = 32,
):
    """Dynamically distribute G128 tiles after one shared A8 pack."""
    counter_ptr = counter_ptr.to(tl.pointer_type(tl.int32))
    _q4_decode_asym_g128_stealing_sdot_kernel(
        lhs_packed_ptr,
        rhs_packed_ptr,
        out_ptr,
        counter_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        STEAL_CHUNK=STEAL_CHUNK,
        LHS_PARTITIONED=False,
    )


@triton.jit
def _q4_fused_decode_asym_g128_stealing_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    STEAL_CHUNK: tl.constexpr = 32,
):
    """Pack A8 per worker and dynamically distribute G128 output tiles."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_row = workspace_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, K // 32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )

    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    counter_ptr = (workspace_bytes + output_byte_offset - 64).to(
        tl.pointer_type(tl.int32)
    )
    _q4_decode_asym_g128_stealing_sdot_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        counter_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        STEAL_CHUNK=STEAL_CHUNK,
    )


@triton.jit
def _q4_decode_asym_g32_compact_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    WEIGHTED: tl.constexpr = False,
):
    """G32 decode with one scale-weighted correction per N4 tile."""
    row = tl.program_id(0)
    encoded_partition = tl.program_id(1)
    partition = encoded_partition & 31 if WEIGHTED else encoded_partition
    partitions = tl.num_programs(1)
    if WEIGHTED:
        # The CPU launcher encodes an exact tile interval in otherwise-unused
        # program-id bits.  Low five bits retain a unique scratch row; the
        # remaining y bits and z carry begin/end offsets.  This lets one
        # launch balance Apple CPU clusters without adding a bounds tensor or
        # changing the public Q4 op ABI.
        local_begin = range_begin + (encoded_partition >> 5)
        local_end = tl.minimum(range_end, range_begin + tl.program_id(2))
    else:
        tile_count = range_end - range_begin
        tiles_per_partition = (tile_count + partitions - 1) // partitions
        local_begin = range_begin + partition * tiles_per_partition
        local_end = tl.minimum(range_end, local_begin + tiles_per_partition)
    groups32: tl.constexpr = K // 32
    rhs_group_stride: tl.constexpr = 72
    rhs_tile_stride: tl.constexpr = groups32 * rhs_group_stride + 16
    lhs_row_stride: tl.constexpr = 8 + K
    output_lanes = tl.arange(0, 4)
    lhs_bytes = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    lhs_row_ptr = lhs_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    lhs_scale = tl.load(lhs_row_ptr.to(tl.pointer_type(tl.float32)))
    zero_point = tl.load(lhs_row_ptr + 4).to(tl.int8).to(tl.int32)
    lhs_data_ptr = lhs_row_ptr + 8

    for tile in range(local_begin, local_end):
        result = tl.zeros((4,), dtype=tl.float32)
        rhs_tile_ptr = rhs_packed_ptr + tile * rhs_tile_stride
        rhs_group_ptr = rhs_tile_ptr
        for group in tl.range(0, groups32, loop_unroll_factor=UNROLL):
            rhs_scale = tl.load(
                rhs_group_ptr.to(tl.pointer_type(tl.bfloat16))
                + output_lanes
            ).to(tl.float32)
            dot_scaled16 = _q4_decode_asym_g128_k32_dot(
                lhs_data_ptr + group * 32,
                rhs_group_ptr + 8,
            )
            result += dot_scaled16.to(tl.float32) * rhs_scale
            rhs_group_ptr += rhs_group_stride

        weighted_sum_scaled16 = tl.load(
            (rhs_tile_ptr + groups32 * rhs_group_stride).to(
                tl.pointer_type(tl.float32)
            )
            + output_lanes
        )
        result -= zero_point.to(tl.float32) * weighted_sum_scaled16
        result *= lhs_scale * (1.0 / 16.0)
        tl.store(
            out_ptr + row * N + tile * 4 + output_lanes,
            result.to(tl.bfloat16),
        )


@triton.jit
def _q4_decode_asym_g32_compact_sdot_flat_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    PARTITIONS: tl.constexpr,
    UNROLL: tl.constexpr = 1,
):
    """Flat-grid form for a matrix launch following a shared A8 pack."""
    program = tl.program_id(0)
    row = program // PARTITIONS
    partition = program % PARTITIONS
    tile_count = range_end - range_begin
    tiles_per_partition = (tile_count + PARTITIONS - 1) // PARTITIONS
    local_begin = range_begin + partition * tiles_per_partition
    local_end = tl.minimum(range_end, local_begin + tiles_per_partition)
    groups32: tl.constexpr = K // 32
    rhs_group_stride: tl.constexpr = 72
    rhs_tile_stride: tl.constexpr = groups32 * rhs_group_stride + 16
    lhs_row_stride: tl.constexpr = 8 + K
    output_lanes = tl.arange(0, 4)
    lhs_bytes = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    lhs_row_ptr = lhs_bytes + row * lhs_row_stride
    lhs_scale = tl.load(lhs_row_ptr.to(tl.pointer_type(tl.float32)))
    zero_point = tl.load(lhs_row_ptr + 4).to(tl.int8).to(tl.int32)
    lhs_data_ptr = lhs_row_ptr + 8

    for tile in range(local_begin, local_end):
        result = tl.zeros((4,), dtype=tl.float32)
        rhs_tile_ptr = rhs_packed_ptr + tile * rhs_tile_stride
        rhs_group_ptr = rhs_tile_ptr
        for group in tl.range(0, groups32, loop_unroll_factor=UNROLL):
            rhs_scale = tl.load(
                rhs_group_ptr.to(tl.pointer_type(tl.bfloat16))
                + output_lanes
            ).to(tl.float32)
            dot_scaled16 = _q4_decode_asym_g128_k32_dot(
                lhs_data_ptr + group * 32,
                rhs_group_ptr + 8,
            )
            result += dot_scaled16.to(tl.float32) * rhs_scale
            rhs_group_ptr += rhs_group_stride

        weighted_sum_scaled16 = tl.load(
            (rhs_tile_ptr + groups32 * rhs_group_stride).to(
                tl.pointer_type(tl.float32)
            )
            + output_lanes
        )
        result -= zero_point.to(tl.float32) * weighted_sum_scaled16
        result *= lhs_scale * (1.0 / 16.0)
        tl.store(
            out_ptr + row * N + tile * 4 + output_lanes,
            result.to(tl.bfloat16),
        )


@triton.jit
def _q4_fused_decode_asym_g32_compact_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
):
    """KleidiAI-compatible activation pack plus compact G32 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups32: tl.constexpr = K // 32
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_row = workspace_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_decode_asym_g32_compact_sdot_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
    )


@triton.jit
def _q4_fused_decode_asym_g32_compact_weighted_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
):
    """Compact G32 GEMV with launcher-provided heterogeneous tile ranges."""
    row = tl.program_id(0)
    encoded_partition = tl.program_id(1)
    partition = encoded_partition & 31
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups32: tl.constexpr = K // 32
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_row = workspace_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_decode_asym_g32_compact_sdot_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        WEIGHTED=True,
    )


@triton.jit
def _q4_decode_asym_g32_compact_stealing_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    counter_ptr,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    STEAL_CHUNK: tl.constexpr = 32,
):
    """Evaluate compact G32 GEMV using a shared N4 work counter."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    groups32: tl.constexpr = K // 32
    lhs_row_stride: tl.constexpr = 8 + K
    lhs_bytes = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    scratch_row = lhs_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    rhs_group_stride: tl.constexpr = 72
    rhs_tile_stride: tl.constexpr = groups32 * rhs_group_stride + 16
    output_lanes = tl.arange(0, 4)
    lhs_scale = tl.load(scratch_row.to(tl.pointer_type(tl.float32)))
    lhs_zero_point = tl.load(scratch_row + 4).to(tl.int8).to(tl.int32)
    lhs_data_ptr = scratch_row + 8

    local_begin = range_begin + tl.atomic_add(counter_ptr, STEAL_CHUNK)
    while local_begin < range_end:
        local_end = tl.minimum(range_end, local_begin + STEAL_CHUNK)
        for tile in range(local_begin, local_end):
            result = tl.zeros((4,), dtype=tl.float32)
            rhs_tile_ptr = rhs_packed_ptr + tile * rhs_tile_stride
            rhs_group_ptr = rhs_tile_ptr
            for group in tl.range(
                0, groups32, loop_unroll_factor=UNROLL
            ):
                rhs_scale = tl.load(
                    rhs_group_ptr.to(tl.pointer_type(tl.bfloat16))
                    + output_lanes
                ).to(tl.float32)
                dot_scaled16 = _q4_decode_asym_g128_k32_dot(
                    lhs_data_ptr + group * 32,
                    rhs_group_ptr + 8,
                )
                result += dot_scaled16.to(tl.float32) * rhs_scale
                rhs_group_ptr += rhs_group_stride

            weighted_sum_scaled16 = tl.load(
                (rhs_tile_ptr + groups32 * rhs_group_stride).to(
                    tl.pointer_type(tl.float32)
                )
                + output_lanes
            )
            result -= (
                lhs_zero_point.to(tl.float32) * weighted_sum_scaled16
            )
            result *= lhs_scale * (1.0 / 16.0)
            tl.store(
                out_ptr + row * N + tile * 4 + output_lanes,
                result.to(tl.bfloat16),
            )
        local_begin = range_begin + tl.atomic_add(counter_ptr, STEAL_CHUNK)


@triton.jit
def _q4_fused_decode_asym_g32_compact_stealing_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    STEAL_CHUNK: tl.constexpr = 32,
):
    """Compact G32 GEMV with in-launch dynamic N4 tile assignment.

    Each OpenMP worker still packs its own activation row, but claims a
    contiguous group of output tiles whenever it finishes the previous one.
    This avoids predicting which Apple CPU cluster a migratable worker will
    occupy for the whole kernel.  The launcher reserves the 64 bytes directly
    before the output for the shared int32 counter.
    """
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_row = workspace_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, K // 32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )

    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    counter_ptr = (workspace_bytes + output_byte_offset - 64).to(
        tl.pointer_type(tl.int32)
    )
    _q4_decode_asym_g32_compact_stealing_sdot_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        counter_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        STEAL_CHUNK=STEAL_CHUNK,
    )


@triton.jit
def _q4_fused_decode_asym_g32_compact_pair_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs0_packed_ptr,
    rhs1_packed_ptr,
    output_byte_offset,
    stride_xm,
    K: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    UNROLL0: tl.constexpr = 1,
    UNROLL1: tl.constexpr = 1,
    STEALING: tl.constexpr = False,
    STEAL_CHUNK: tl.constexpr = 32,
):
    """Pack one decode activation and evaluate two compact G32 matrices.

    Qwen GDN projects qkvz and ba from the same hidden state.  Keeping both
    matrices in one program removes one OpenMP team launch and one activation
    quantization while preserving each matrix's packed checkpoint layout.
    The native entry point currently restricts this kernel to one decode row.
    """
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_row = workspace_bytes + (
        row * partitions + partition
    ) * lhs_row_stride
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, K // 32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )

    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    if STEALING:
        counter_ptr = (workspace_bytes + output_byte_offset - 64).to(
            tl.pointer_type(tl.int32)
        )
        _q4_decode_asym_g32_compact_stealing_sdot_kernel(
            workspace_bytes,
            rhs0_packed_ptr,
            out_ptr,
            counter_ptr,
            0,
            N0 // 4,
            K=K,
            N=N0,
            UNROLL=UNROLL0,
            STEAL_CHUNK=STEAL_CHUNK,
        )
    else:
        _q4_decode_asym_g32_compact_sdot_kernel(
            workspace_bytes,
            rhs0_packed_ptr,
            out_ptr,
            0,
            N0 // 4,
            K=K,
            N=N0,
            UNROLL=UNROLL0,
        )
    _q4_decode_asym_g32_compact_sdot_kernel(
        workspace_bytes,
        rhs1_packed_ptr,
        out_ptr + N0,
        0,
        N1 // 4,
        K=K,
        N=N1,
        UNROLL=UNROLL1,
    )


@triton.jit
def _pack_lhs_qai8dxp_asym_decode_kai_kernel(
    x_ptr,
    lhs_packed_ptr,
    stride_xm,
    K: tl.constexpr,
):
    """Pack one KAI-compatible A8 row once for all decode partitions."""
    row = tl.program_id(0)
    source_row = x_ptr + row * stride_xm
    lhs_bytes = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    packed = lhs_bytes + row * (8 + K)
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    tl.store(
        packed.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1), scale
    )
    tl.store(packed + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, K // 32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            packed + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )


@triton.jit
def _q4_pack_swiglu_asym_kai_kernel(
    joined_ptr,
    scratch_ptr,
    lhs_packed_ptr,
    stride_joined_m,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse exact BF16 SwiGLU materialization with shared KAI A8 packing."""
    row = tl.program_id(0)
    joined = joined_ptr + row * stride_joined_m
    scratch = scratch_ptr + row * K
    lhs_bytes = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))
    packed = lhs_bytes + row * (8 + K)
    lanes = tl.arange(0, BLOCK_SIZE)
    row_min = tl.full((1,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((1,), -3.4028234663852886e38, tl.float32)
    for base in tl.range(0, K, BLOCK_SIZE, loop_unroll_factor=1):
        offsets = base + lanes
        gate = tl.load(joined + offsets).to(tl.float32)
        up = tl.load(joined + K + offsets).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        value = (silu * up).to(tl.bfloat16)
        tl.store(scratch + offsets, value)
        value_f32 = value.to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(value_f32, axis=0))
        row_max = tl.maximum(row_max, tl.max(value_f32, axis=0))

    scale, quant_multiplier, zero_point = (
        _q4_kai_asymmetric_qparams_from_minmax(row_min, row_max)
    )
    tl.store(
        packed.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1), scale
    )
    tl.store(packed + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, K // 32, loop_unroll_factor=1):
        source = scratch + group * 32
        _q4_store_token_asymmetric_k32_kai(
            packed + 8 + group * 32,
            source,
            quant_multiplier,
            zero_point,
        )


@triton.jit
def _q4_fused_decode_asym_g32_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr = 1,
):
    """KleidiAI ``qai8dxp_f32`` activation pack plus G32 Q4 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups32: tl.constexpr = K // 32
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = (
        _q4_token_asymmetric_qparams_kai_f32(source_row, K=K)
    )
    scratch_base = (row * partitions + partition) * lhs_row_stride
    scratch_row = workspace_bytes + scratch_base
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_decode_asym_sdot_kai_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        LHS_PARTITIONED=True,
        LHS_COMPACT=True,
        LHS_KAI=True,
    )


@triton.jit
def _q4_fused_decode_asym_sdot_kai_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Single-entry token-asymmetric activation pack and Q4 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups: tl.constexpr = K // 32
    lhs_group_stride: tl.constexpr = 36
    source_row = x_ptr + row * stride_xm
    scale, inv_scale, zero_point = _q4_token_asymmetric_qparams_bf16(
        source_row, K=K
    )
    scratch_base = (
        (row * partitions + partition) * groups * lhs_group_stride
    )
    scratch_row = workspace_bytes + scratch_base
    for group in tl.range(0, groups, loop_unroll_factor=1):
        blob = scratch_row + group * lhs_group_stride
        tl.store(
            blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 1), scale
        )
        tl.store(blob + 2 + tl.arange(0, 1), zero_point)
        _q4_store_token_asymmetric_k32(
            blob + 4,
            source_row + group * 32,
            inv_scale,
            zero_point,
        )
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_decode_asym_sdot_kai_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        LHS_PARTITIONED=True,
    )


@triton.jit
def _q4_fused_decode_sdot_kai_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Replicated-LHS single-entry Q4 GEMV for CPU program partitions.

    CPU Triton programs have no cross-program shared-memory barrier.  Each N
    partition therefore emits the small K32 activation blob into private
    scratch before entering the unchanged SDOT GEMV body.  This deliberately
    repeats a few microseconds of activation work to remove an entire Python
    launcher round trip while keeping all computation compiler-visible.
    """
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups: tl.constexpr = K // 32
    lhs_group_stride: tl.constexpr = 34
    scratch_base = (
        (row * partitions + partition) * groups * lhs_group_stride
    )
    scratch_row = workspace_bytes + scratch_base
    scratch_row_i8 = scratch_row.to(tl.pointer_type(tl.int8))
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    lanes = tl.arange(0, 8)

    for group in tl.range(0, groups, loop_unroll_factor=1):
        x_base = x_ptr + row * stride_xm + group * 32
        # Four native K8 values fit without a hot-loop spill on the current
        # AArch64 lowering.  Reuse them after absmax instead of loading the
        # activation row a second time for quantization.
        values0 = tl.load(x_base + lanes)
        values1 = tl.load(x_base + 8 + lanes)
        values2 = tl.load(x_base + 16 + lanes)
        values3 = tl.load(x_base + 24 + lanes)
        absmax = _q4_bf16_values_absmax(
            values0, values1, values2, values3
        )
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = group * lhs_group_stride
        tl.store(
            (scratch_row + blob_base).to(
                tl.pointer_type(tl.float16)
            ),
            scale.to(tl.float16),
        )
        _q4_store_quantized_bf16_values(
            scratch_row_i8 + blob_base + 2,
            values0,
            values1,
            values2,
            values3,
            inv_scale,
        )

    _q4_decode_sdot_kai_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        LHS_PARTITIONED=True,
    )


@triton.jit(do_not_specialize=["rms_eps"])
def _q4_fused_rmsnorm_decode_sdot_kai_kernel(
    x_ptr,
    rms_weight_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    rms_eps,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    NORM_TILE: tl.constexpr,
):
    """Single-entry Qwen RMSNorm, activation quantization, and Q4 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    groups: tl.constexpr = K // 32
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    lhs_group_stride: tl.constexpr = 34
    source_row = x_ptr + row * stride_xm

    sum_sq = tl.zeros((1,), dtype=tl.float32)
    norm_lanes = tl.arange(0, NORM_TILE)
    for start in tl.range(0, K, NORM_TILE, loop_unroll_factor=1):
        values = tl.load(source_row + start + norm_lanes).to(tl.float32)
        sum_sq += tl.sum(values * values, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / K + rms_eps)

    scratch_base = (
        (row * partitions + partition) * groups * lhs_group_stride
    )
    scratch_row = workspace_bytes + scratch_base
    scratch_row_i8 = scratch_row.to(tl.pointer_type(tl.int8))
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )

    for group in tl.range(0, groups, loop_unroll_factor=1):
        x_base = source_row + group * 32
        weight_base = rms_weight_ptr + group * 32
        values0 = _q4_rmsnorm_bf16_values8(
            x_base, weight_base, 0, rrms
        )
        values1 = _q4_rmsnorm_bf16_values8(
            x_base, weight_base, 8, rrms
        )
        values2 = _q4_rmsnorm_bf16_values8(
            x_base, weight_base, 16, rrms
        )
        values3 = _q4_rmsnorm_bf16_values8(
            x_base, weight_base, 24, rrms
        )
        absmax = _q4_bf16_values_absmax(
            values0, values1, values2, values3
        )
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = group * lhs_group_stride
        tl.store(
            (scratch_row + blob_base).to(
                tl.pointer_type(tl.float16)
            ),
            scale.to(tl.float16),
        )
        _q4_store_quantized_bf16_values(
            scratch_row_i8 + blob_base + 2,
            values0,
            values1,
            values2,
            values3,
            inv_scale,
        )

    _q4_decode_sdot_kai_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        LHS_PARTITIONED=True,
    )


@triton.jit
def _q4_apply_owned_qk_rmsnorm(
    out_ptr,
    qk_weight_ptr,
    range_begin,
    range_end,
    qk_eps,
    N: tl.constexpr,
    Q_ELEMENTS: tl.constexpr,
    K_ELEMENTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NORM_TILE: tl.constexpr,
):
    """Normalize the Q/K heads produced by this output partition in place.

    The specialized launcher only chooses partitions whose output boundaries
    are whole-head aligned.  A program can therefore reduce and update every
    head it owns without a cross-program barrier or an overlapping store.
    """
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    tile_count = range_end - range_begin
    tiles_per_partition = (tile_count + partitions - 1) // partitions
    local_begin = range_begin + partition * tiles_per_partition
    local_end = tl.minimum(range_end, local_begin + tiles_per_partition)
    element_begin = local_begin * 4
    element_end = tl.minimum(local_end * 4, Q_ELEMENTS + K_ELEMENTS)
    head_begin = element_begin // HEAD_DIM
    head_end = element_end // HEAD_DIM
    q_heads: tl.constexpr = Q_ELEMENTS // HEAD_DIM
    lanes = tl.arange(0, NORM_TILE)
    output_row = out_ptr + row * N

    # This remains a rolled loop in LLVM.  It does not unroll all Q/K heads
    # into one wide SSA value, and each head retires before the next begins.
    for head in range(head_begin, head_end):
        head_ptr = output_row + head * HEAD_DIM
        weight_offset = tl.where(head < q_heads, 0, HEAD_DIM)
        weight_ptr = qk_weight_ptr + weight_offset
        sum_sq = tl.zeros((1,), dtype=tl.float32)
        for start in range(0, HEAD_DIM, NORM_TILE):
            values = tl.load(head_ptr + start + lanes).to(tl.float32)
            sum_sq += tl.sum(values * values, axis=0)
        rrms = 1.0 / tl.sqrt(sum_sq / HEAD_DIM + qk_eps)
        for start in range(0, HEAD_DIM, NORM_TILE):
            values = tl.load(head_ptr + start + lanes).to(tl.float32)
            weights = tl.load(weight_ptr + start + lanes).to(tl.float32)
            # Match Qwen3RMSNorm's BF16 boundary before weight multiply.
            normalized = (values * rrms).to(tl.bfloat16).to(tl.float32)
            tl.store(
                head_ptr + start + lanes,
                (normalized * weights).to(tl.bfloat16),
            )


@triton.jit(do_not_specialize=["rms_eps", "qk_eps"])
def _q4_fused_rmsnorm_qk_norm_decode_sdot_kai_kernel(
    x_ptr,
    rms_weight_ptr,
    qk_weight_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    rms_eps,
    qk_eps,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    Q_ELEMENTS: tl.constexpr,
    K_ELEMENTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NORM_TILE: tl.constexpr,
):
    """Input RMSNorm + joined QKV projection + Q/K head RMSNorm."""
    _q4_fused_rmsnorm_decode_sdot_kai_kernel(
        x_ptr,
        rms_weight_ptr,
        workspace_ptr,
        rhs_packed_ptr,
        output_byte_offset,
        stride_xm,
        range_begin,
        range_end,
        rms_eps,
        K=K,
        N=N,
        UNROLL=UNROLL,
        NORM_TILE=NORM_TILE,
    )
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    _q4_apply_owned_qk_rmsnorm(
        out_ptr,
        qk_weight_ptr,
        range_begin,
        range_end,
        qk_eps,
        N=N,
        Q_ELEMENTS=Q_ELEMENTS,
        K_ELEMENTS=K_ELEMENTS,
        HEAD_DIM=HEAD_DIM,
        NORM_TILE=NORM_TILE,
    )


@triton.jit(do_not_specialize=["rms_eps"])
def _q4_fused_add_rmsnorm_decode_sdot_kai_kernel(
    input_ptr,
    residual_ptr,
    rms_weight_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    summed_byte_offset,
    output_byte_offset,
    stride_xm,
    range_begin,
    range_end,
    rms_eps,
    K: tl.constexpr,
    N: tl.constexpr,
    UNROLL: tl.constexpr,
    NORM_TILE: tl.constexpr,
):
    """Single-entry residual add, RMSNorm, quantization, and Q4 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    rows = tl.num_programs(0)
    partitions = tl.num_programs(1)
    groups: tl.constexpr = K // 32
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    lhs_group_stride: tl.constexpr = 34
    input_row = input_ptr + row * stride_xm
    residual_row = residual_ptr + row * stride_xm
    summed_ptr = (workspace_bytes + summed_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    # Partition-major layout makes partition 0's rows one contiguous tensor.
    # That slice becomes the returned residual without any shared publication
    # store from inside this parallel kernel.
    summed_row = summed_ptr + (partition * rows + row) * K

    # Every partition owns a private BF16 residual sum.  This is required:
    # publishing the shared residual before all programs finish reading its
    # old value would introduce a cross-program race in the absence of a CPU
    # Triton barrier.
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    norm_lanes = tl.arange(0, NORM_TILE)
    for start in tl.range(0, K, NORM_TILE, loop_unroll_factor=1):
        input_values = tl.load(
            input_row + start + norm_lanes
        ).to(tl.float32)
        residual_values = tl.load(
            residual_row + start + norm_lanes
        ).to(tl.float32)
        summed = (input_values + residual_values).to(tl.bfloat16)
        tl.store(summed_row + start + norm_lanes, summed)
        summed_fp32 = summed.to(tl.float32)
        sum_sq += tl.sum(summed_fp32 * summed_fp32, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / K + rms_eps)

    scratch_base = (
        (row * partitions + partition) * groups * lhs_group_stride
    )
    scratch_row = workspace_bytes + scratch_base
    scratch_row_i8 = scratch_row.to(tl.pointer_type(tl.int8))
    out_ptr = (workspace_bytes + output_byte_offset).to(
        tl.pointer_type(tl.bfloat16)
    )
    for group in tl.range(0, groups, loop_unroll_factor=1):
        summed_base = summed_row + group * 32
        weight_base = rms_weight_ptr + group * 32
        values0 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 0, rrms
        )
        values1 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 8, rrms
        )
        values2 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 16, rrms
        )
        values3 = _q4_rmsnorm_stored_bf16_values8(
            summed_base, weight_base, 24, rrms
        )
        absmax = _q4_bf16_values_absmax(
            values0, values1, values2, values3
        )
        scale = absmax / 127.0
        inv_scale = tl.where(scale != 0.0, 1.0 / scale, 0.0)
        blob_base = group * lhs_group_stride
        tl.store(
            (scratch_row + blob_base).to(
                tl.pointer_type(tl.float16)
            ),
            scale.to(tl.float16),
        )
        _q4_store_quantized_bf16_values(
            scratch_row_i8 + blob_base + 2,
            values0,
            values1,
            values2,
            values3,
            inv_scale,
        )

    _q4_decode_sdot_kai_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        range_begin,
        range_end,
        K=K,
        N=N,
        UNROLL=UNROLL,
        LHS_PARTITIONED=True,
    )


@triton.jit
def _pack_lhs_qsi8d32p_asym_panel4_kernel(
    x_ptr,
    lhs_packed_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Pack four token-wise asymmetric rows for the I8MM prefill path."""
    panel = tl.program_id(0)
    row0 = panel * 4
    groups: tl.constexpr = K // 32
    panel_group_stride: tl.constexpr = 144
    source0 = tl.minimum(row0, M - 1)
    source1 = tl.minimum(row0 + 1, M - 1)
    source2 = tl.minimum(row0 + 2, M - 1)
    source3 = tl.minimum(row0 + 3, M - 1)
    x0 = x_ptr + source0 * stride_xm
    x1 = x_ptr + source1 * stride_xm
    x2 = x_ptr + source2 * stride_xm
    x3 = x_ptr + source3 * stride_xm
    scale0, inv0, zp0 = _q4_token_asymmetric_qparams_bf16(x0, K=K)
    scale1, inv1, zp1 = _q4_token_asymmetric_qparams_bf16(x1, K=K)
    scale2, inv2, zp2 = _q4_token_asymmetric_qparams_bf16(x2, K=K)
    scale3, inv3, zp3 = _q4_token_asymmetric_qparams_bf16(x3, K=K)
    scales = tl.join(
        tl.join(scale0, scale2), tl.join(scale1, scale3)
    ).reshape((4,))
    zero_points = tl.join(
        tl.join(zp0, zp2), tl.join(zp1, zp3)
    ).reshape((4,))
    lanes = tl.arange(0, 8)
    store_lanes = tl.arange(0, 32)
    for group in tl.range(0, groups, loop_unroll_factor=1):
        blob = lhs_packed_ptr + (panel * groups + group) * panel_group_stride
        tl.store(blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4), scales)
        tl.store(blob + 8 + tl.arange(0, 4), zero_points)
        data_base = blob + 16
        for offset in tl.static_range(0, 32, 8):
            quant0 = _quantize_token_asymmetric_i8(
                tl.load(x0 + group * 32 + offset + lanes), inv0, zp0
            )
            quant1 = _quantize_token_asymmetric_i8(
                tl.load(x1 + group * 32 + offset + lanes), inv1, zp1
            )
            quant2 = _quantize_token_asymmetric_i8(
                tl.load(x2 + group * 32 + offset + lanes), inv2, zp2
            )
            quant3 = _quantize_token_asymmetric_i8(
                tl.load(x3 + group * 32 + offset + lanes), inv3, zp3
            )
            quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
            quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
            quant0123 = tl.join(quant01, quant23).permute(1, 0).reshape((32,))
            tl.store(data_base + offset * 4 + store_lanes, quant0123)


@triton.jit
def _pack_lhs_qsi8d128p_asym_panel4_kernel(
    x_ptr,
    lhs_packed_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Compact token-asymmetric panel4 pack for the native-G128 path.

    One 16-byte panel header owns four token scales/zero points; K32 I8MM
    data follows contiguously.  Quantization parameters are reduced for all
    four rows together instead of emitting four independent scalar loops.
    """
    panel = tl.program_id(0)
    rows = panel * 4 + tl.arange(0, 4)
    source_rows = tl.minimum(rows, M - 1)
    lanes16 = tl.arange(0, 16)
    row_min = tl.full((4,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((4,), -3.4028234663852886e38, tl.float32)
    for start in tl.range(0, K, 16, loop_unroll_factor=1):
        values = tl.load(
            x_ptr
            + source_rows[:, None] * stride_xm
            + start
            + lanes16[None, :]
        ).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values, axis=1))
        row_max = tl.maximum(row_max, tl.max(values, axis=1))
    scales, inv_scales, zero_points = _q4_asymmetric_qparams_from_minmax(
        row_min, row_max
    )
    panel_stride: tl.constexpr = 16 + 4 * K
    panel_base = lhs_packed_ptr + panel * panel_stride
    tl.store(
        panel_base.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4),
        scales,
    )
    tl.store(panel_base + 8 + tl.arange(0, 4), zero_points)

    lanes8 = tl.arange(0, 8)
    store_lanes = tl.arange(0, 32)
    groups32: tl.constexpr = K // 32
    data_base = panel_base + 16
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        for offset in tl.static_range(0, 32, 8):
            values = tl.load(
                x_ptr
                + source_rows[:, None] * stride_xm
                + group * 32
                + offset
                + lanes8[None, :]
            )
            quantized = _quantize_token_asymmetric_i8(
                values,
                inv_scales[:, None],
                zero_points[:, None],
            )
            tl.store(
                data_base + group * 128 + offset * 4 + store_lanes,
                quantized.reshape((32,)),
            )


@triton.jit
def _pack_lhs_qai8dxp_asym_panel4_kernel(
    x_ptr,
    lhs_packed_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Panel4 pack with KleidiAI's FP32 asymmetric activation semantics."""
    panel = tl.program_id(0)
    rows = panel * 4 + tl.arange(0, 4)
    source_rows = tl.minimum(rows, M - 1)
    lanes16 = tl.arange(0, 16)
    row_min = tl.full((4,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((4,), -3.4028234663852886e38, tl.float32)
    for start in tl.range(0, K, 16, loop_unroll_factor=1):
        values = tl.load(
            x_ptr
            + source_rows[:, None] * stride_xm
            + start
            + lanes16[None, :]
        ).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values, axis=1))
        row_max = tl.maximum(row_max, tl.max(values, axis=1))
    scales, multipliers, zero_points = (
        _q4_kai_asymmetric_qparams_from_minmax(row_min, row_max)
    )
    panel_stride: tl.constexpr = 32 + 4 * K
    panel_base = lhs_packed_ptr + panel * panel_stride
    tl.store(
        panel_base.to(tl.pointer_type(tl.float32)) + tl.arange(0, 4),
        scales,
    )
    tl.store(panel_base + 16 + tl.arange(0, 4), zero_points)

    lanes8 = tl.arange(0, 8)
    store_lanes = tl.arange(0, 32)
    groups32: tl.constexpr = K // 32
    data_base = panel_base + 32
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        for offset in tl.static_range(0, 32, 8):
            values = tl.load(
                x_ptr
                + source_rows[:, None] * stride_xm
                + group * 32
                + offset
                + lanes8[None, :]
            )
            quantized = _quantize_kai_asymmetric_i8(
                values,
                multipliers[:, None],
                zero_points[:, None],
            )
            tl.store(
                data_base + group * 128 + offset * 4 + store_lanes,
                quantized.reshape((32,)),
            )


@triton.jit
def _q4_pack_swiglu_asym_panel4_kai_kernel(
    joined_ptr,
    scratch_ptr,
    lhs_packed_ptr,
    M,
    stride_joined_m,
    K: tl.constexpr,
):
    """Fuse exact BF16 SwiGLU with the Prefill panel4 A8 pack.

    The first pass materializes each BF16 activation once while collecting
    the four row ranges.  The second pass quantizes that scratch into the
    same KAI-compatible layout as ``_pack_lhs_qai8dxp_asym_panel4_kernel``.
    This replaces separate ATen SiLU and multiply launches without evaluating
    the exponential twice.
    """
    panel = tl.program_id(0)
    rows = panel * 4 + tl.arange(0, 4)
    source_rows = tl.minimum(rows, M - 1)
    joined_rows = joined_ptr + source_rows[:, None] * stride_joined_m
    scratch_rows = scratch_ptr + rows[:, None] * K
    lanes16 = tl.arange(0, 16)
    row_min = tl.full((4,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((4,), -3.4028234663852886e38, tl.float32)
    for start in tl.range(0, K, 16, loop_unroll_factor=1):
        gate = tl.load(joined_rows + start + lanes16[None, :]).to(tl.float32)
        up = tl.load(joined_rows + K + start + lanes16[None, :]).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        # Match vLLM's two BF16 ATen operations exactly: SiLU rounds before
        # multiplication, then the product rounds again.
        silu = silu.to(tl.bfloat16).to(tl.float32)
        values = (silu * up).to(tl.bfloat16)
        tl.store(scratch_rows + start + lanes16[None, :], values)
        values_f32 = values.to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values_f32, axis=1))
        row_max = tl.maximum(row_max, tl.max(values_f32, axis=1))

    scales, multipliers, zero_points = (
        _q4_kai_asymmetric_qparams_from_minmax(row_min, row_max)
    )
    panel_stride: tl.constexpr = 32 + 4 * K
    panel_base = lhs_packed_ptr + panel * panel_stride
    tl.store(
        panel_base.to(tl.pointer_type(tl.float32)) + tl.arange(0, 4),
        scales,
    )
    tl.store(panel_base + 16 + tl.arange(0, 4), zero_points)

    lanes8 = tl.arange(0, 8)
    store_lanes = tl.arange(0, 32)
    data_base = panel_base + 32
    for group in tl.range(0, K // 32, loop_unroll_factor=1):
        for offset in tl.static_range(0, 32, 8):
            values = tl.load(
                scratch_rows
                + group * 32
                + offset
                + lanes8[None, :]
            )
            quantized = _quantize_kai_asymmetric_i8(
                values,
                multipliers[:, None],
                zero_points[:, None],
            )
            tl.store(
                data_base + group * 128 + offset * 4 + store_lanes,
                quantized.reshape((32,)),
            )


@triton.jit
def _q4_prefill_i8mm_kai_kernel(
    lhs_data_ptr,
    lhs_scale_ptr,
    rhs_data_ptr,
    rhs_scale_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute one M4/M8/M12/M16 by N4 tile from exact KAI blobs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_n: tl.constexpr = 4
    groups: tl.constexpr = K // 32
    num_panels: tl.constexpr = BLOCK_M // 4
    cols = pid_n * block_n + tl.arange(0, block_n)
    lanes_m4 = tl.arange(0, 4)

    result0 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 8:
        result1 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 12:
        result2 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 16:
        result3 = tl.zeros((4, block_n), tl.float32)

    for group in range(0, groups):
        rhs_blob_base = (pid_n * groups + group) * 72
        packed_flat = tl.load(
            rhs_data_ptr + rhs_blob_base + 8 + tl.arange(0, 64)
        )
        packed = packed_flat.reshape((2, block_n, 8)).permute(
            0, 2, 1
        ).reshape((16, block_n))
        weight_low = (packed << 4).to(tl.int8)
        weight_high = (packed & 0xF0).to(tl.int8)
        weight = tl.join(weight_low, weight_high).permute(
            0, 2, 1
        ).reshape((32, block_n))

        lhs_blob_base = (
            pid_m * num_panels * groups + group
        ) * 136
        lhs_panel_stride: tl.constexpr = groups * 136
        rhs_scale = tl.load(
            rhs_scale_ptr + rhs_blob_base // 2 + tl.arange(0, block_n)
        ).to(tl.float32)

        lhs0_seq = tl.load(
            lhs_data_ptr + lhs_blob_base + 8 + tl.arange(0, 128)
        ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        lhs0 = lhs0_seq.reshape((4, 2, 16)).permute(
            0, 2, 1
        ).reshape((4, 32))
        scale0 = tl.load(
            lhs_scale_ptr + lhs_blob_base // 2 + lanes_m4
        ).to(tl.float32)
        dot0 = tl.dot(lhs0, weight, out_dtype=tl.int32)
        result0 += (
            dot0.to(tl.float32) * (1.0 / 16.0)
            * scale0[:, None] * rhs_scale[None, :]
        )

        if BLOCK_M >= 8:
            lhs1_base = lhs_blob_base + lhs_panel_stride
            lhs1_seq = tl.load(
                lhs_data_ptr + lhs1_base + 8 + tl.arange(0, 128)
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs1 = lhs1_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            scale1 = tl.load(
                lhs_scale_ptr + lhs1_base // 2 + lanes_m4
            ).to(tl.float32)
            dot1 = tl.dot(lhs1, weight, out_dtype=tl.int32)
            result1 += (
                dot1.to(tl.float32) * (1.0 / 16.0)
                * scale1[:, None] * rhs_scale[None, :]
            )

        if BLOCK_M >= 12:
            lhs2_base = lhs_blob_base + 2 * lhs_panel_stride
            lhs2_seq = tl.load(
                lhs_data_ptr + lhs2_base + 8 + tl.arange(0, 128)
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs2 = lhs2_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            scale2 = tl.load(
                lhs_scale_ptr + lhs2_base // 2 + lanes_m4
            ).to(tl.float32)
            dot2 = tl.dot(lhs2, weight, out_dtype=tl.int32)
            result2 += (
                dot2.to(tl.float32) * (1.0 / 16.0)
                * scale2[:, None] * rhs_scale[None, :]
            )

        if BLOCK_M >= 16:
            lhs3_base = lhs_blob_base + 3 * lhs_panel_stride
            lhs3_seq = tl.load(
                lhs_data_ptr + lhs3_base + 8 + tl.arange(0, 128)
            ).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs3 = lhs3_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            scale3 = tl.load(
                lhs_scale_ptr + lhs3_base // 2 + lanes_m4
            ).to(tl.float32)
            dot3 = tl.dot(lhs3, weight, out_dtype=tl.int32)
            result3 += (
                dot3.to(tl.float32) * (1.0 / 16.0)
                * scale3[:, None] * rhs_scale[None, :]
            )

    output_row = pid_m * BLOCK_M
    tl.store(
        out_ptr + (output_row + lanes_m4)[:, None] * N + cols[None, :],
        result0.to(tl.bfloat16),
    )
    if BLOCK_M >= 8:
        tl.store(
            out_ptr
            + (output_row + 4 + lanes_m4)[:, None] * N
            + cols[None, :],
            result1.to(tl.bfloat16),
        )
    if BLOCK_M >= 12:
        tl.store(
            out_ptr
            + (output_row + 8 + lanes_m4)[:, None] * N
            + cols[None, :],
            result2.to(tl.bfloat16),
        )
    if BLOCK_M >= 16:
        tl.store(
            out_ptr
            + (output_row + 12 + lanes_m4)[:, None] * N
            + cols[None, :],
            result3.to(tl.bfloat16),
        )


@triton.jit
def _q4_prefill_asym_i8mm_kai_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LHS_KAI: tl.constexpr = False,
):
    """I8MM prefill over the compact token-asymmetric panel4 layout.

    Activation scale/zero-point metadata belongs to a token, not to a weight
    group. One compact header (16 bytes for BF16 fake quantization or 32
    bytes for ``qai8dxp_f32``) is shared by all K32 slices in a panel. This
    avoids repeating metadata and keeps the activation packer's live range
    small enough for register allocation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_n: tl.constexpr = 4
    groups: tl.constexpr = K // 32
    num_panels: tl.constexpr = BLOCK_M // 4
    cols = pid_n * block_n + tl.arange(0, block_n)
    lanes_m4 = tl.arange(0, 4)
    result0 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 8:
        result1 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 12:
        result2 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 16:
        result3 = tl.zeros((4, block_n), tl.float32)

    for group in range(0, groups):
        rhs_blob = rhs_packed_ptr + (pid_n * groups + group) * 80
        packed_flat = tl.load(rhs_blob + 8 + tl.arange(0, 64))
        packed = packed_flat.reshape((2, block_n, 8)).permute(
            0, 2, 1
        ).reshape((16, block_n))
        weight_low = (packed << 4).to(tl.int8)
        weight_high = (packed & 0xF0).to(tl.int8)
        weight = tl.join(weight_low, weight_high).permute(
            0, 2, 1
        ).reshape((32, block_n))
        rhs_scale = tl.load(
            rhs_blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4)
        ).to(tl.float32)
        rhs_sum = tl.load(
            (rhs_blob + 72).to(tl.pointer_type(tl.int16)) + tl.arange(0, 4)
        ).to(tl.int32)
        lhs_panel_stride: tl.constexpr = (32 if LHS_KAI else 16) + 4 * K
        lhs_data_offset: tl.constexpr = 32 if LHS_KAI else 16
        lhs_zp_offset: tl.constexpr = 16 if LHS_KAI else 8
        lhs0_panel = (
            lhs_packed_ptr + pid_m * num_panels * lhs_panel_stride
        )
        lhs0_data = lhs0_panel + lhs_data_offset + group * 128

        lhs0_seq = (
            tl.load(lhs0_data + tl.arange(0, 128))
            .to(tl.int8)
            .reshape((4, 4, 8))
            .permute(1, 0, 2)
            .reshape((4, 32))
        )
        lhs0 = lhs0_seq.reshape((4, 2, 16)).permute(
            0, 2, 1
        ).reshape((4, 32))
        if LHS_KAI:
            scale0 = tl.load(
                lhs0_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
            )
        else:
            scale0 = tl.load(
                lhs0_panel.to(tl.pointer_type(tl.bfloat16)) + lanes_m4
            ).to(tl.float32)
        zp0 = tl.load(
            lhs0_panel + lhs_zp_offset + lanes_m4
        ).to(tl.int8).to(tl.int32)
        dot0 = tl.dot(lhs0, weight, out_dtype=tl.int32)
        corrected0 = dot0 - zp0[:, None] * rhs_sum[None, :]
        contribution0 = corrected0.to(tl.float32) * (1.0 / 16.0)
        contribution0 *= scale0[:, None]
        contribution0 *= rhs_scale[None, :]
        result0 += contribution0

        if BLOCK_M >= 8:
            lhs1_panel = lhs0_panel + lhs_panel_stride
            lhs1_seq = (
                tl.load(
                    lhs1_panel
                    + lhs_data_offset
                    + group * 128
                    + tl.arange(0, 128)
                )
                .to(tl.int8)
                .reshape((4, 4, 8))
                .permute(1, 0, 2)
                .reshape((4, 32))
            )
            lhs1 = lhs1_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            if LHS_KAI:
                scale1 = tl.load(
                    lhs1_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
                )
            else:
                scale1 = tl.load(
                    lhs1_panel.to(tl.pointer_type(tl.bfloat16)) + lanes_m4
                ).to(tl.float32)
            zp1 = tl.load(
                lhs1_panel + lhs_zp_offset + lanes_m4
            ).to(tl.int8).to(tl.int32)
            dot1 = tl.dot(lhs1, weight, out_dtype=tl.int32)
            corrected1 = dot1 - zp1[:, None] * rhs_sum[None, :]
            contribution1 = corrected1.to(tl.float32) * (1.0 / 16.0)
            contribution1 *= scale1[:, None]
            contribution1 *= rhs_scale[None, :]
            result1 += contribution1

        if BLOCK_M >= 12:
            lhs2_panel = lhs0_panel + 2 * lhs_panel_stride
            lhs2_seq = (
                tl.load(
                    lhs2_panel
                    + lhs_data_offset
                    + group * 128
                    + tl.arange(0, 128)
                )
                .to(tl.int8)
                .reshape((4, 4, 8))
                .permute(1, 0, 2)
                .reshape((4, 32))
            )
            lhs2 = lhs2_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            if LHS_KAI:
                scale2 = tl.load(
                    lhs2_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
                )
            else:
                scale2 = tl.load(
                    lhs2_panel.to(tl.pointer_type(tl.bfloat16)) + lanes_m4
                ).to(tl.float32)
            zp2 = tl.load(
                lhs2_panel + lhs_zp_offset + lanes_m4
            ).to(tl.int8).to(tl.int32)
            dot2 = tl.dot(lhs2, weight, out_dtype=tl.int32)
            corrected2 = dot2 - zp2[:, None] * rhs_sum[None, :]
            contribution2 = corrected2.to(tl.float32) * (1.0 / 16.0)
            contribution2 *= scale2[:, None]
            contribution2 *= rhs_scale[None, :]
            result2 += contribution2

        if BLOCK_M >= 16:
            lhs3_panel = lhs0_panel + 3 * lhs_panel_stride
            lhs3_seq = (
                tl.load(
                    lhs3_panel
                    + lhs_data_offset
                    + group * 128
                    + tl.arange(0, 128)
                )
                .to(tl.int8)
                .reshape((4, 4, 8))
                .permute(1, 0, 2)
                .reshape((4, 32))
            )
            lhs3 = lhs3_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            if LHS_KAI:
                scale3 = tl.load(
                    lhs3_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
                )
            else:
                scale3 = tl.load(
                    lhs3_panel.to(tl.pointer_type(tl.bfloat16)) + lanes_m4
                ).to(tl.float32)
            zp3 = tl.load(
                lhs3_panel + lhs_zp_offset + lanes_m4
            ).to(tl.int8).to(tl.int32)
            dot3 = tl.dot(lhs3, weight, out_dtype=tl.int32)
            corrected3 = dot3 - zp3[:, None] * rhs_sum[None, :]
            contribution3 = corrected3.to(tl.float32) * (1.0 / 16.0)
            contribution3 *= scale3[:, None]
            contribution3 *= rhs_scale[None, :]
            result3 += contribution3

    output_row = pid_m * BLOCK_M
    tl.store(
        out_ptr + (output_row + lanes_m4)[:, None] * N + cols[None, :],
        result0.to(tl.bfloat16),
    )
    if BLOCK_M >= 8:
        tl.store(
            out_ptr + (output_row + 4 + lanes_m4)[:, None] * N + cols[None, :],
            result1.to(tl.bfloat16),
        )
    if BLOCK_M >= 12:
        tl.store(
            out_ptr + (output_row + 8 + lanes_m4)[:, None] * N + cols[None, :],
            result2.to(tl.bfloat16),
        )
    if BLOCK_M >= 16:
        tl.store(
            out_ptr + (output_row + 12 + lanes_m4)[:, None] * N + cols[None, :],
            result3.to(tl.bfloat16),
        )


@triton.jit
def _q4_prefill_asym_g32_compact_i8mm_kai_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """I8MM prefill over compact G32 RHS tiles.

    The dot products still consume one BF16 scale and 64 packed Q4 bytes per
    K32 slice.  Activation zero-point correction is linear across K, so the
    per-slice INT16 sums are replaced by one scale-weighted FP32 footer per
    N4 tile.  Activation metadata is loaded once per panel instead of once
    per K32 slice.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_n: tl.constexpr = 4
    groups: tl.constexpr = K // 32
    num_panels: tl.constexpr = BLOCK_M // 4
    rhs_tile_stride: tl.constexpr = groups * 72 + 16
    lhs_panel_stride: tl.constexpr = 32 + 4 * K
    cols = pid_n * block_n + tl.arange(0, block_n)
    lanes_m4 = tl.arange(0, 4)
    lhs0_panel = lhs_packed_ptr + pid_m * num_panels * lhs_panel_stride
    result0 = tl.zeros((4, block_n), tl.float32)

    if BLOCK_M >= 8:
        lhs1_panel = lhs0_panel + lhs_panel_stride
        result1 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 12:
        lhs2_panel = lhs0_panel + 2 * lhs_panel_stride
        result2 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 16:
        lhs3_panel = lhs0_panel + 3 * lhs_panel_stride
        result3 = tl.zeros((4, block_n), tl.float32)

    rhs_tile = rhs_packed_ptr + pid_n * rhs_tile_stride
    for group in range(0, groups):
        rhs_blob = rhs_tile + group * 72
        packed_flat = tl.load(rhs_blob + 8 + tl.arange(0, 64))
        packed = packed_flat.reshape((2, block_n, 8)).permute(
            0, 2, 1
        ).reshape((16, block_n))
        weight_low = (packed << 4).to(tl.int8)
        weight_high = (packed & 0xF0).to(tl.int8)
        weight = tl.join(weight_low, weight_high).permute(
            0, 2, 1
        ).reshape((32, block_n))
        rhs_scale = tl.load(
            rhs_blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4)
        ).to(tl.float32)

        lhs0_seq = (
            tl.load(lhs0_panel + 32 + group * 128 + tl.arange(0, 128))
            .to(tl.int8)
            .reshape((4, 4, 8))
            .permute(1, 0, 2)
            .reshape((4, 32))
        )
        lhs0 = lhs0_seq.reshape((4, 2, 16)).permute(
            0, 2, 1
        ).reshape((4, 32))
        result0 += (
            tl.dot(lhs0, weight, out_dtype=tl.int32).to(tl.float32)
            * rhs_scale[None, :]
        )

        if BLOCK_M >= 8:
            lhs1_seq = (
                tl.load(lhs1_panel + 32 + group * 128 + tl.arange(0, 128))
                .to(tl.int8)
                .reshape((4, 4, 8))
                .permute(1, 0, 2)
                .reshape((4, 32))
            )
            lhs1 = lhs1_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            result1 += (
                tl.dot(lhs1, weight, out_dtype=tl.int32).to(tl.float32)
                * rhs_scale[None, :]
            )
        if BLOCK_M >= 12:
            lhs2_seq = (
                tl.load(lhs2_panel + 32 + group * 128 + tl.arange(0, 128))
                .to(tl.int8)
                .reshape((4, 4, 8))
                .permute(1, 0, 2)
                .reshape((4, 32))
            )
            lhs2 = lhs2_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            result2 += (
                tl.dot(lhs2, weight, out_dtype=tl.int32).to(tl.float32)
                * rhs_scale[None, :]
            )
        if BLOCK_M >= 16:
            lhs3_seq = (
                tl.load(lhs3_panel + 32 + group * 128 + tl.arange(0, 128))
                .to(tl.int8)
                .reshape((4, 4, 8))
                .permute(1, 0, 2)
                .reshape((4, 32))
            )
            lhs3 = lhs3_seq.reshape((4, 2, 16)).permute(
                0, 2, 1
            ).reshape((4, 32))
            result3 += (
                tl.dot(lhs3, weight, out_dtype=tl.int32).to(tl.float32)
                * rhs_scale[None, :]
            )

    weighted_sum_scaled16 = tl.load(
        (rhs_tile + groups * 72).to(tl.pointer_type(tl.float32))
        + tl.arange(0, 4)
    )
    scale0 = tl.load(
        lhs0_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
    )
    zp0 = tl.load(lhs0_panel + 16 + lanes_m4).to(tl.int8).to(tl.int32)
    result0 -= zp0[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
    result0 *= scale0[:, None] * (1.0 / 16.0)
    if BLOCK_M >= 8:
        scale1 = tl.load(
            lhs1_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
        )
        zp1 = tl.load(lhs1_panel + 16 + lanes_m4).to(tl.int8).to(tl.int32)
        result1 -= (
            zp1[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
        )
        result1 *= scale1[:, None] * (1.0 / 16.0)
    if BLOCK_M >= 12:
        scale2 = tl.load(
            lhs2_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
        )
        zp2 = tl.load(lhs2_panel + 16 + lanes_m4).to(tl.int8).to(tl.int32)
        result2 -= (
            zp2[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
        )
        result2 *= scale2[:, None] * (1.0 / 16.0)
    if BLOCK_M >= 16:
        scale3 = tl.load(
            lhs3_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
        )
        zp3 = tl.load(lhs3_panel + 16 + lanes_m4).to(tl.int8).to(tl.int32)
        result3 -= (
            zp3[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
        )
        result3 *= scale3[:, None] * (1.0 / 16.0)

    output_row = pid_m * BLOCK_M
    tl.store(
        out_ptr + (output_row + lanes_m4)[:, None] * N + cols[None, :],
        result0.to(tl.bfloat16),
    )
    if BLOCK_M >= 8:
        tl.store(
            out_ptr + (output_row + 4 + lanes_m4)[:, None] * N + cols[None, :],
            result1.to(tl.bfloat16),
        )
    if BLOCK_M >= 12:
        tl.store(
            out_ptr + (output_row + 8 + lanes_m4)[:, None] * N + cols[None, :],
            result2.to(tl.bfloat16),
        )
    if BLOCK_M >= 16:
        tl.store(
            out_ptr + (output_row + 12 + lanes_m4)[:, None] * N + cols[None, :],
            result3.to(tl.bfloat16),
        )


@triton.jit
def _q4_prefill_asym_g32_compact_i8mm_kai_m8n8_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    OUTPUT_STRIDE: tl.constexpr,
    K: tl.constexpr,
):
    """Experimental M8xN8 compact prefill tile.

    Compact weights are stored as independent N4 tiles.  Pairing two of
    those tiles in one program reuses each unpacked M8 activation panel for
    both dot products while keeping the same 64-FP32-output accumulator
    footprint as the production M16xN4 kernel.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_n: tl.constexpr = 4
    groups: tl.constexpr = K // 32
    rhs_tile_stride: tl.constexpr = groups * 72 + 16
    lhs_panel_stride: tl.constexpr = 32 + 4 * K
    lanes_m4 = tl.arange(0, 4)
    cols0 = pid_n * 8 + tl.arange(0, block_n)
    cols1 = cols0 + 4

    lhs0_panel = lhs_packed_ptr + pid_m * 2 * lhs_panel_stride
    lhs1_panel = lhs0_panel + lhs_panel_stride
    rhs0_tile = rhs_packed_ptr + (pid_n * 2) * rhs_tile_stride
    rhs1_tile = rhs0_tile + rhs_tile_stride
    result00 = tl.zeros((4, block_n), tl.float32)
    result01 = tl.zeros((4, block_n), tl.float32)
    result10 = tl.zeros((4, block_n), tl.float32)
    result11 = tl.zeros((4, block_n), tl.float32)

    for group in range(0, groups):
        rhs0_blob = rhs0_tile + group * 72
        packed0_flat = tl.load(rhs0_blob + 8 + tl.arange(0, 64))
        packed0 = packed0_flat.reshape((2, block_n, 8)).permute(
            0, 2, 1
        ).reshape((16, block_n))
        weight0_low = (packed0 << 4).to(tl.int8)
        weight0_high = (packed0 & 0xF0).to(tl.int8)
        weight0 = tl.join(weight0_low, weight0_high).permute(
            0, 2, 1
        ).reshape((32, block_n))
        rhs0_scale = tl.load(
            rhs0_blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4)
        ).to(tl.float32)

        rhs1_blob = rhs1_tile + group * 72
        packed1_flat = tl.load(rhs1_blob + 8 + tl.arange(0, 64))
        packed1 = packed1_flat.reshape((2, block_n, 8)).permute(
            0, 2, 1
        ).reshape((16, block_n))
        weight1_low = (packed1 << 4).to(tl.int8)
        weight1_high = (packed1 & 0xF0).to(tl.int8)
        weight1 = tl.join(weight1_low, weight1_high).permute(
            0, 2, 1
        ).reshape((32, block_n))
        rhs1_scale = tl.load(
            rhs1_blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4)
        ).to(tl.float32)

        lhs0_seq = (
            tl.load(lhs0_panel + 32 + group * 128 + tl.arange(0, 128))
            .to(tl.int8)
            .reshape((4, 4, 8))
            .permute(1, 0, 2)
            .reshape((4, 32))
        )
        lhs0 = lhs0_seq.reshape((4, 2, 16)).permute(
            0, 2, 1
        ).reshape((4, 32))
        result00 += (
            tl.dot(lhs0, weight0, out_dtype=tl.int32).to(tl.float32)
            * rhs0_scale[None, :]
        )
        result01 += (
            tl.dot(lhs0, weight1, out_dtype=tl.int32).to(tl.float32)
            * rhs1_scale[None, :]
        )

        lhs1_seq = (
            tl.load(lhs1_panel + 32 + group * 128 + tl.arange(0, 128))
            .to(tl.int8)
            .reshape((4, 4, 8))
            .permute(1, 0, 2)
            .reshape((4, 32))
        )
        lhs1 = lhs1_seq.reshape((4, 2, 16)).permute(
            0, 2, 1
        ).reshape((4, 32))
        result10 += (
            tl.dot(lhs1, weight0, out_dtype=tl.int32).to(tl.float32)
            * rhs0_scale[None, :]
        )
        result11 += (
            tl.dot(lhs1, weight1, out_dtype=tl.int32).to(tl.float32)
            * rhs1_scale[None, :]
        )

    weighted_sum0_scaled16 = tl.load(
        (rhs0_tile + groups * 72).to(tl.pointer_type(tl.float32))
        + tl.arange(0, 4)
    )
    weighted_sum1_scaled16 = tl.load(
        (rhs1_tile + groups * 72).to(tl.pointer_type(tl.float32))
        + tl.arange(0, 4)
    )
    scale0 = tl.load(
        lhs0_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
    )
    scale1 = tl.load(
        lhs1_panel.to(tl.pointer_type(tl.float32)) + lanes_m4
    )
    zp0 = tl.load(lhs0_panel + 16 + lanes_m4).to(tl.int8).to(tl.int32)
    zp1 = tl.load(lhs1_panel + 16 + lanes_m4).to(tl.int8).to(tl.int32)
    result00 -= zp0[:, None].to(tl.float32) * weighted_sum0_scaled16[None, :]
    result01 -= zp0[:, None].to(tl.float32) * weighted_sum1_scaled16[None, :]
    result10 -= zp1[:, None].to(tl.float32) * weighted_sum0_scaled16[None, :]
    result11 -= zp1[:, None].to(tl.float32) * weighted_sum1_scaled16[None, :]
    result00 *= scale0[:, None] * (1.0 / 16.0)
    result01 *= scale0[:, None] * (1.0 / 16.0)
    result10 *= scale1[:, None] * (1.0 / 16.0)
    result11 *= scale1[:, None] * (1.0 / 16.0)

    output_row = pid_m * 8
    tl.store(
        out_ptr
        + (output_row + lanes_m4)[:, None] * OUTPUT_STRIDE
        + cols0[None, :],
        result00.to(tl.bfloat16),
    )
    tl.store(
        out_ptr
        + (output_row + lanes_m4)[:, None] * OUTPUT_STRIDE
        + cols1[None, :],
        result01.to(tl.bfloat16),
    )
    tl.store(
        out_ptr
        + (output_row + 4 + lanes_m4)[:, None] * OUTPUT_STRIDE
        + cols0[None, :],
        result10.to(tl.bfloat16),
    )
    tl.store(
        out_ptr
        + (output_row + 4 + lanes_m4)[:, None] * OUTPUT_STRIDE
        + cols1[None, :],
        result11.to(tl.bfloat16),
    )


@triton.jit
def _q4_load_g128_k32_i8mm_weight(rhs_data_ptr):
    """Unpack one K32 slice while preserving the I8MM lowering algebra."""
    block_n: tl.constexpr = 4
    packed_flat = tl.load(rhs_data_ptr + tl.arange(0, 64))
    packed = packed_flat.reshape((2, block_n, 8)).permute(
        0, 2, 1
    ).reshape((16, block_n))
    weight_low = (packed << 4).to(tl.int8)
    weight_high = (packed & 0xF0).to(tl.int8)
    return tl.join(weight_low, weight_high).permute(
        0, 2, 1
    ).reshape((32, block_n))


@triton.jit
def _q4_load_g128_k32_i8mm_lhs(lhs_blob):
    """Load one panel4 K32 slice in the layout recognized by the dot pass."""
    lhs_seq = tl.load(lhs_blob + tl.arange(0, 128)).to(tl.int8).reshape(
        (4, 4, 8)
    ).permute(1, 0, 2).reshape((4, 32))
    return lhs_seq.reshape((4, 2, 16)).permute(
        0, 2, 1
    ).reshape((4, 32))


@triton.jit
def _q4_prefill_asym_g128_i8mm_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LHS_KAI: tl.constexpr = False,
    SUBGROUP_UNROLL: tl.constexpr = 1,
):
    """G128 Q4 prefill: four K32 SMMLA bodies per scale/correction."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_n: tl.constexpr = 4
    groups128: tl.constexpr = K // 128
    num_panels: tl.constexpr = BLOCK_M // 4
    cols = pid_n * block_n + tl.arange(0, block_n)
    lanes_m4 = tl.arange(0, 4)
    result0 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 8:
        result1 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 12:
        result2 = tl.zeros((4, block_n), tl.float32)
    if BLOCK_M >= 16:
        result3 = tl.zeros((4, block_n), tl.float32)

    if LHS_KAI:
        lhs_panel_stride: tl.constexpr = 32 + 4 * K
        lhs_data_offset: tl.constexpr = 32
        lhs_zp_offset: tl.constexpr = 16
    else:
        lhs_panel_stride: tl.constexpr = 16 + 4 * K
        lhs_data_offset: tl.constexpr = 16
        lhs_zp_offset: tl.constexpr = 8
    lhs_row_base = lhs_packed_ptr + pid_m * num_panels * lhs_panel_stride
    if LHS_KAI:
        scale0 = tl.load(
            lhs_row_base.to(tl.pointer_type(tl.float32)) + lanes_m4
        )
    else:
        scale0 = tl.load(
            lhs_row_base.to(tl.pointer_type(tl.bfloat16)) + lanes_m4
        ).to(tl.float32)
    zp0 = tl.load(
        lhs_row_base + lhs_zp_offset + lanes_m4
    ).to(tl.int8).to(tl.int32)
    if BLOCK_M >= 8:
        if LHS_KAI:
            scale1 = tl.load(
                (lhs_row_base + lhs_panel_stride).to(
                    tl.pointer_type(tl.float32)
                ) + lanes_m4
            )
        else:
            scale1 = tl.load(
                (lhs_row_base + lhs_panel_stride).to(
                    tl.pointer_type(tl.bfloat16)
                ) + lanes_m4
            ).to(tl.float32)
        zp1 = tl.load(
            lhs_row_base + lhs_panel_stride + lhs_zp_offset + lanes_m4
        ).to(tl.int8).to(tl.int32)
    if BLOCK_M >= 12:
        if LHS_KAI:
            scale2 = tl.load(
                (lhs_row_base + 2 * lhs_panel_stride).to(
                    tl.pointer_type(tl.float32)
                ) + lanes_m4
            )
        else:
            scale2 = tl.load(
                (lhs_row_base + 2 * lhs_panel_stride).to(
                    tl.pointer_type(tl.bfloat16)
                ) + lanes_m4
            ).to(tl.float32)
        zp2 = tl.load(
            lhs_row_base
            + 2 * lhs_panel_stride
            + lhs_zp_offset
            + lanes_m4
        ).to(tl.int8).to(tl.int32)
    if BLOCK_M >= 16:
        if LHS_KAI:
            scale3 = tl.load(
                (lhs_row_base + 3 * lhs_panel_stride).to(
                    tl.pointer_type(tl.float32)
                ) + lanes_m4
            )
        else:
            scale3 = tl.load(
                (lhs_row_base + 3 * lhs_panel_stride).to(
                    tl.pointer_type(tl.bfloat16)
                ) + lanes_m4
            ).to(tl.float32)
        zp3 = tl.load(
            lhs_row_base
            + 3 * lhs_panel_stride
            + lhs_zp_offset
            + lanes_m4
        ).to(tl.int8).to(tl.int32)

    rhs_tile_stride: tl.constexpr = groups128 * 264 + 16
    rhs_tile_ptr = rhs_packed_ptr + pid_n * rhs_tile_stride

    for group128 in tl.range(0, groups128, loop_unroll_factor=1):
        rhs_blob = rhs_tile_ptr + group128 * 264
        rhs_scale = tl.load(
            rhs_blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4)
        ).to(tl.float32)
        dot0 = tl.zeros((4, block_n), dtype=tl.int32)
        if BLOCK_M >= 8:
            dot1 = tl.zeros((4, block_n), dtype=tl.int32)
        if BLOCK_M >= 12:
            dot2 = tl.zeros((4, block_n), dtype=tl.int32)
        if BLOCK_M >= 16:
            dot3 = tl.zeros((4, block_n), dtype=tl.int32)
        for subgroup in tl.range(
            0, 4, loop_unroll_factor=SUBGROUP_UNROLL
        ):
            group32 = group128 * 4 + subgroup
            weight = _q4_load_g128_k32_i8mm_weight(
                rhs_blob + 8 + subgroup * 64
            )
            lhs0_blob = lhs_row_base + lhs_data_offset + group32 * 128
            dot0 += tl.dot(
                _q4_load_g128_k32_i8mm_lhs(lhs0_blob),
                weight,
                out_dtype=tl.int32,
            )
            if BLOCK_M >= 8:
                dot1 += tl.dot(
                    _q4_load_g128_k32_i8mm_lhs(
                        lhs0_blob + lhs_panel_stride
                    ),
                    weight,
                    out_dtype=tl.int32,
                )
            if BLOCK_M >= 12:
                dot2 += tl.dot(
                    _q4_load_g128_k32_i8mm_lhs(
                        lhs0_blob + 2 * lhs_panel_stride
                    ),
                    weight,
                    out_dtype=tl.int32,
                )
            if BLOCK_M >= 16:
                dot3 += tl.dot(
                    _q4_load_g128_k32_i8mm_lhs(
                        lhs0_blob + 3 * lhs_panel_stride
                    ),
                    weight,
                    out_dtype=tl.int32,
                )

        result0 += dot0.to(tl.float32) * rhs_scale[None, :]
        if BLOCK_M >= 8:
            result1 += dot1.to(tl.float32) * rhs_scale[None, :]
        if BLOCK_M >= 12:
            result2 += dot2.to(tl.float32) * rhs_scale[None, :]
        if BLOCK_M >= 16:
            result3 += dot3.to(tl.float32) * rhs_scale[None, :]

    weighted_sum_scaled16 = tl.load(
        (rhs_tile_ptr + groups128 * 264).to(tl.pointer_type(tl.float32))
        + tl.arange(0, 4)
    )
    result0 -= zp0[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
    result0 *= scale0[:, None] * (1.0 / 16.0)
    if BLOCK_M >= 8:
        result1 -= zp1[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
        result1 *= scale1[:, None] * (1.0 / 16.0)
    if BLOCK_M >= 12:
        result2 -= zp2[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
        result2 *= scale2[:, None] * (1.0 / 16.0)
    if BLOCK_M >= 16:
        result3 -= zp3[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
        result3 *= scale3[:, None] * (1.0 / 16.0)

    output_row = pid_m * BLOCK_M
    tl.store(
        out_ptr + (output_row + lanes_m4)[:, None] * N + cols[None, :],
        result0.to(tl.bfloat16),
    )
    if BLOCK_M >= 8:
        tl.store(
            out_ptr + (output_row + 4 + lanes_m4)[:, None] * N + cols[None, :],
            result1.to(tl.bfloat16),
        )
    if BLOCK_M >= 12:
        tl.store(
            out_ptr + (output_row + 8 + lanes_m4)[:, None] * N + cols[None, :],
            result2.to(tl.bfloat16),
        )
    if BLOCK_M >= 16:
        tl.store(
            out_ptr + (output_row + 12 + lanes_m4)[:, None] * N + cols[None, :],
            result3.to(tl.bfloat16),
        )


@triton.jit
def _q4_prefill_asym_g128_i8mm_kai_m12_k32_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """M12 KAI tile that retires each recognized M4 dot immediately."""
    pid_n = tl.program_id(1)
    block_n: tl.constexpr = 4
    groups128: tl.constexpr = K // 128
    lhs_panel_stride: tl.constexpr = 32 + 4 * K
    lhs_data_offset: tl.constexpr = 32
    cols = pid_n * block_n + tl.arange(0, block_n)
    lanes_m4 = tl.arange(0, 4)
    result0 = tl.zeros((4, block_n), tl.float32)
    result1 = tl.zeros((4, block_n), tl.float32)
    result2 = tl.zeros((4, block_n), tl.float32)

    scale0 = tl.load(
        lhs_packed_ptr.to(tl.pointer_type(tl.float32)) + lanes_m4
    )
    scale1 = tl.load(
        (lhs_packed_ptr + lhs_panel_stride).to(tl.pointer_type(tl.float32))
        + lanes_m4
    )
    scale2 = tl.load(
        (lhs_packed_ptr + 2 * lhs_panel_stride).to(tl.pointer_type(tl.float32))
        + lanes_m4
    )
    zp0 = tl.load(lhs_packed_ptr + 16 + lanes_m4).to(tl.int8).to(tl.int32)
    zp1 = tl.load(
        lhs_packed_ptr + lhs_panel_stride + 16 + lanes_m4
    ).to(tl.int8).to(tl.int32)
    zp2 = tl.load(
        lhs_packed_ptr + 2 * lhs_panel_stride + 16 + lanes_m4
    ).to(tl.int8).to(tl.int32)

    rhs_tile_stride: tl.constexpr = groups128 * 264 + 16
    rhs_tile_ptr = rhs_packed_ptr + pid_n * rhs_tile_stride
    for group128 in tl.range(0, groups128, loop_unroll_factor=1):
        rhs_blob = rhs_tile_ptr + group128 * 264
        rhs_scale = tl.load(
            rhs_blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4)
        ).to(tl.float32)
        for subgroup in tl.range(0, 4, loop_unroll_factor=1):
            group32 = group128 * 4 + subgroup
            weight = _q4_load_g128_k32_i8mm_weight(
                rhs_blob + 8 + subgroup * 64
            )
            lhs0_blob = lhs_packed_ptr + lhs_data_offset + group32 * 128
            dot0 = tl.dot(
                _q4_load_g128_k32_i8mm_lhs(lhs0_blob),
                weight,
                out_dtype=tl.int32,
            )
            result0 += dot0.to(tl.float32) * rhs_scale[None, :]
            dot1 = tl.dot(
                _q4_load_g128_k32_i8mm_lhs(
                    lhs0_blob + lhs_panel_stride
                ),
                weight,
                out_dtype=tl.int32,
            )
            result1 += dot1.to(tl.float32) * rhs_scale[None, :]
            dot2 = tl.dot(
                _q4_load_g128_k32_i8mm_lhs(
                    lhs0_blob + 2 * lhs_panel_stride
                ),
                weight,
                out_dtype=tl.int32,
            )
            result2 += dot2.to(tl.float32) * rhs_scale[None, :]

    weighted_sum_scaled16 = tl.load(
        (rhs_tile_ptr + groups128 * 264).to(tl.pointer_type(tl.float32))
        + tl.arange(0, 4)
    )
    result0 -= zp0[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
    result0 *= scale0[:, None] * (1.0 / 16.0)
    result1 -= zp1[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
    result1 *= scale1[:, None] * (1.0 / 16.0)
    result2 -= zp2[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
    result2 *= scale2[:, None] * (1.0 / 16.0)

    tl.store(
        out_ptr + lanes_m4[:, None] * N + cols[None, :],
        result0.to(tl.bfloat16),
    )
    tl.store(
        out_ptr + (4 + lanes_m4)[:, None] * N + cols[None, :],
        result1.to(tl.bfloat16),
    )
    tl.store(
        out_ptr + (8 + lanes_m4)[:, None] * N + cols[None, :],
        result2.to(tl.bfloat16),
    )
