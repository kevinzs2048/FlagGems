"""Target-aware vector scheduling constants for ordinary Triton CPU kernels."""

from __future__ import annotations

import ctypes
import os
import platform


def _detect_vector_bits() -> int:
    override = os.environ.get("FLAGGEMS_ARM_VECTOR_BITS")
    if override:
        bits = int(override)
        if bits < 128 or bits % 128:
            raise ValueError(
                "FLAGGEMS_ARM_VECTOR_BITS must be a positive multiple of 128"
            )
        return bits

    # Apple M-series and baseline AArch64 expose 128-bit Advanced SIMD.
    # Linux SVE can be wider and its process vector length is queryable.
    if platform.system() == "Linux" and platform.machine() in (
        "aarch64",
        "arm64",
    ):
        try:
            pr_sve_get_vl = 51
            pr_sve_vl_len_mask = 0xFFFF
            value = ctypes.CDLL(None).prctl(pr_sve_get_vl, 0, 0, 0, 0)
            if value >= 0:
                return max(128, (value & pr_sve_vl_len_mask) * 8)
        except (AttributeError, OSError):
            pass
    return 128


VECTOR_BITS = _detect_vector_bits()
FP32_LANES = VECTOR_BITS // 32


def fp32_tile(unroll: int = 1) -> int:
    """Return a power-of-two FP32 tile spanning ``unroll`` native vectors."""
    lanes = FP32_LANES * unroll
    if lanes <= 0 or lanes & (lanes - 1):
        raise ValueError("Triton block dimensions must be powers of two")
    return lanes


# Reductions need independent accumulators to hide latency. Elementwise
# transforms use one architectural vector and rely on the rolled SCF loop.
REDUCTION_TILE = fp32_tile(unroll=4)
ELEMENTWISE_TILE = fp32_tile()
# A rolled elementwise loop benefits from a few independent vectors without
# materializing the GPU-style 128/256-element SSA tile.  Four vectors is also
# the scheduling unit used by the reduction kernels and scales with SVE VL.
# Cap the logical tile at 64 values.  On 128-bit NEON/SVE this supplies enough
# independent vectors to hide load/store latency; on wider SVE the same tile
# naturally needs fewer architectural vectors and avoids register pressure.
ELEMENTWISE_ROLLED_TILE = min(64, fp32_tile(unroll=16))
# Transcendentals lower to vector-library calls on current AArch64 targets.
# Four 128-bit vectors hide latency without keeping a large live set across
# those calls; wider SVE targets cover the same 16 values with fewer vectors.
NONLINEAR_ROLLED_TILE = min(16, fp32_tile(unroll=4))

# Below this size the cost of entering the CPU backend's parallel program-grid
# path is larger than the saved memory-loop time on the tested Arm cores.
# Keep it overridable for targets with a different thread-launch/grain balance.
SINGLE_PROGRAM_MAX_ELEMENTS = int(
    os.environ.get("FLAGGEMS_ARM_SINGLE_PROGRAM_MAX_ELEMENTS", "262144")
)
