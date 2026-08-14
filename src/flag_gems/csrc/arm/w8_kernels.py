"""Importable FlagGems W8 entry points used by the C++ JIT router.

Operator implementations live in the standalone FlagGems repository.  This
module is intentionally only an ABI-stable import shim for libtriton_jit.
"""

from flag_gems.runtime.backend._arm.int8.w8_kernels import (
    _pack_lhs_qai8dxp_bf16_kernel,
    _pack_lhs_qai8dxp_bf16_mr4_kernel,
    _pack_lhs_w8_i8mm_kai_kernel,
    _pack_lhs_w8_i8mm_kai_vllm_trunc_kernel,
    _quantize_bf16_w8_rne_kernel,
    _quantize_bf16_w8_vllm_trunc_kernel,
    _w8_decode_sdot_kernel,
    _w8_prefill_i8mm_kai_kernel,
    _w8_prefill_i8mm_kai_m12_kernel,
    _w8_prefill_i8mm_kai_short_tail_kernel,
    _w8_qai8dxp_decode_sdot_kernel,
    _w8_qai8dxp_decode_stealing_sdot_kernel,
    _w8_qai8dxp_prefill_i8mm_kernel,
    _w8_qai8dxp_prefill_m12_kernel,
    _w8_qai8dxp_prefill_short_tail_kernel,
)

__all__ = [
    "_pack_lhs_w8_i8mm_kai_kernel",
    "_pack_lhs_w8_i8mm_kai_vllm_trunc_kernel",
    "_quantize_bf16_w8_rne_kernel",
    "_quantize_bf16_w8_vllm_trunc_kernel",
    "_w8_decode_sdot_kernel",
    "_w8_prefill_i8mm_kai_kernel",
    "_w8_prefill_i8mm_kai_m12_kernel",
    "_w8_prefill_i8mm_kai_short_tail_kernel",
    "_pack_lhs_qai8dxp_bf16_kernel",
    "_pack_lhs_qai8dxp_bf16_mr4_kernel",
    "_w8_qai8dxp_decode_sdot_kernel",
    "_w8_qai8dxp_decode_stealing_sdot_kernel",
    "_w8_qai8dxp_prefill_i8mm_kernel",
    "_w8_qai8dxp_prefill_m12_kernel",
    "_w8_qai8dxp_prefill_short_tail_kernel",
]
