"""ARM CPU fused SiLU-and-multiply.

The default decode path is an ordinary Triton program.  A small logical tile
is carried through a rolled loop so LLVM can generate native-width NEON/SVE
math without expanding the full hidden dimension in SSA.  The legacy coarse
TLE runtime call remains available as an explicit diagnostic fallback.
"""


import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:
    from triton.language.extra.cpu.tle_ops import swiglu as _tle_swiglu
except ImportError:
    _tle_swiglu = None

from ..vector_config import FP32_LANES

# None = not yet tried, True = TLE path works, False = fall back to ATen.
_TLE_SWIGLU_OK = None
_SWIGLU_MODE = os.environ.get("GEMS_ARM_SWIGLU_MODE", "auto").lower()
_SWIGLU_TILE = min(16, FP32_LANES * 4)
try:
    _SWIGLU_MIN_ELEMENTS = max(
        0, int(os.environ.get("GEMS_ARM_SWIGLU_MIN_ELEMENTS", "8192"))
    )
except ValueError:
    _SWIGLU_MIN_ELEMENTS = 8192


@triton.jit
def _round_to_nearest_even_i32(value):
    return libdevice.rint(value).to(tl.int32)


@triton.jit
def _sleef_expf_u10_inline(value):
    """SLEEF-u10 exp polynomial expressed as ordinary Triton arithmetic."""
    exponent = _round_to_nearest_even_i32(value * 1.4426950408889634)
    exponent_f = exponent.to(tl.float32)
    reduced = tl.fma(exponent_f, -0.693145751953125, value)
    reduced = tl.fma(
        exponent_f, -1.428606765330187e-6, reduced
    )

    polynomial = tl.full(
        value.shape, 0.00019852761761285365, tl.float32
    )
    polynomial = tl.fma(
        polynomial, reduced, 0.0013930435525253415
    )
    polynomial = tl.fma(
        polynomial, reduced, 0.008333360776305199
    )
    polynomial = tl.fma(
        polynomial, reduced, 0.041666485369205475
    )
    polynomial = tl.fma(
        polynomial, reduced, 0.1666666716337204
    )
    polynomial = tl.fma(polynomial, reduced, 0.5)
    result = 1.0 + tl.fma(
        reduced * reduced, polynomial, reduced
    )

    half_exponent = exponent >> 1
    pow0_bits = (half_exponent + 127) << 23
    pow1_bits = (exponent - half_exponent + 127) << 23
    pow0 = pow0_bits.to(tl.float32, bitcast=True)
    pow1 = pow1_bits.to(tl.float32, bitcast=True)
    result = result * pow0 * pow1
    result = tl.where(value < -104.0, 0.0, result)
    return tl.where(value > 100.0, float("inf"), result)


@triton.jit
def _swiglu_ordinary_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_SIZE)
    full_elements: tl.constexpr = (
        n_elements // BLOCK_SIZE
    ) * BLOCK_SIZE
    # Keep the main body rolled even though the extent is shape-specialized.
    # Specialization removes the impossible masked tail for aligned model
    # dimensions without expanding the hidden dimension in SSA.
    for base in tl.range(
        0, full_elements, BLOCK_SIZE, loop_unroll_factor=1
    ):
        idx = base + lanes
        gate = tl.load(gate_ptr + idx).to(tl.float32)
        up = tl.load(up_ptr + idx).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        # F.silu(BF16) exposes a BF16 intermediate before the multiply.
        silu = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + idx, (silu * up).to(tl.bfloat16))
    if n_elements % BLOCK_SIZE:
        idx = full_elements + lanes
        mask = idx < n_elements
        gate = tl.load(gate_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(
            out_ptr + idx,
            (silu * up).to(tl.bfloat16),
            mask=mask,
        )


@triton.jit
def _swiglu_quantize_w8_rne_kernel(
    gate_ptr,
    up_ptr,
    bf16_ptr,
    q_ptr,
    scale_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize BF16 SwiGLU and W8-RNE quantize it in one launch.

    The first rolled pass preserves the observable BF16 SiLU and multiply
    boundaries while accumulating the exact BF16 magnitude maximum.  The
    second pass reads that BF16 scratch once and applies the same
    127/absmax + round-to-nearest-even contract as the standalone decoder
    quantizer.
    """
    lanes = tl.arange(0, BLOCK_SIZE)
    full_elements: tl.constexpr = (
        n_elements // BLOCK_SIZE
    ) * BLOCK_SIZE
    absmax = tl.zeros((1,), dtype=tl.float32)
    for base in tl.range(
        0, full_elements, BLOCK_SIZE, loop_unroll_factor=1
    ):
        idx = base + lanes
        gate = tl.load(gate_ptr + idx).to(tl.float32)
        up = tl.load(up_ptr + idx).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        value = (silu * up).to(tl.bfloat16)
        tl.store(bf16_ptr + idx, value)
        bits = (
            value.to(tl.uint16, bitcast=True) & 0x7FFF
        ).to(tl.int16)
        block_bits = tl.max(bits, axis=0).to(tl.uint16)
        block_absmax = block_bits.to(
            tl.bfloat16, bitcast=True
        ).to(tl.float32)
        absmax = tl.maximum(absmax, block_absmax)
    if n_elements % BLOCK_SIZE:
        idx = full_elements + lanes
        mask = idx < n_elements
        gate = tl.load(gate_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        silu = gate / (1.0 + _sleef_expf_u10_inline(-gate))
        silu = silu.to(tl.bfloat16).to(tl.float32)
        value = (silu * up).to(tl.bfloat16)
        tl.store(bf16_ptr + idx, value, mask=mask)
        absmax = tl.maximum(
            absmax,
            tl.max(
                tl.where(mask, tl.abs(value.to(tl.float32)), 0.0),
                axis=0,
            ),
        )

    absmax = tl.maximum(absmax, 1.0e-8)
    scale = absmax / 127.0
    inv_scale = 127.0 / absmax
    tl.store(scale_ptr + tl.arange(0, 1), scale)
    for base in tl.range(
        0, n_elements, BLOCK_SIZE, loop_unroll_factor=1
    ):
        idx = base + lanes
        mask = idx < n_elements
        value = tl.load(
            bf16_ptr + idx, mask=mask, other=0.0
        ).to(tl.float32)
        quantized = libdevice.rint(value * inv_scale).to(tl.int8)
        tl.store(q_ptr + idx, quantized, mask=mask)


@triton.jit
def _swiglu_quantize_w8_rne_joined_kernel(
    joined_ptr,
    bf16_ptr,
    q_ptr,
    scale_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Consume adjacent gate/up halves without Python slice dispatches."""
    _swiglu_quantize_w8_rne_kernel(
        joined_ptr,
        joined_ptr + n_elements,
        bf16_ptr,
        q_ptr,
        scale_ptr,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _swiglu_ordinary_joined_kernel(
    joined_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    _swiglu_ordinary_kernel(
        joined_ptr,
        joined_ptr + n_elements,
        out_ptr,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, N: tl.constexpr):
    # One coarse TLE op = the whole SWIGLU (silu(gate) * up over N elements),
    # OMP-parallelized inside the C runtime → 1 kernel launch.
    _tle_swiglu(gate_ptr, up_ptr, out_ptr, N)


def arm_silu_and_mul(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(x1) * x2`` with BF16 decode semantics."""
    global _TLE_SWIGLU_OK
    decode_compatible = (
        x1.dtype == torch.bfloat16
        and x2.dtype == torch.bfloat16
        and x1.shape == x2.shape
        and x1.is_contiguous()
        and x2.is_contiguous()
        and x1.numel() == x1.shape[-1]
    )
    use_ordinary = _SWIGLU_MODE == "ordinary" or (
        _SWIGLU_MODE == "auto"
        and x1.numel() >= _SWIGLU_MIN_ELEMENTS
    )
    if decode_compatible and use_ordinary:
        out = torch.empty_like(x1)
        _swiglu_ordinary_kernel[(1,)](
            x1,
            x2,
            out,
            x1.numel(),
            BLOCK_SIZE=_SWIGLU_TILE,
            num_warps=1,
            num_stages=1,
        )
        return out

    if (
        decode_compatible
        and _SWIGLU_MODE == "tle"
        and _tle_swiglu is not None
        and _TLE_SWIGLU_OK is not False
    ):
        try:
            N = x1.numel()
            out = torch.empty_like(x1)
            _swiglu_kernel[(1,)](x1, x2, out, N=N)
            _TLE_SWIGLU_OK = True
            return out
        except Exception:
            _TLE_SWIGLU_OK = False
    return F.silu(x1) * x2


def arm_silu_and_mul_out(
    x1: torch.Tensor, x2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """ARM CPU fused silu_and_mul with pre-allocated output."""
    result = arm_silu_and_mul(x1, x2)
    out.copy_(result)
    return out
