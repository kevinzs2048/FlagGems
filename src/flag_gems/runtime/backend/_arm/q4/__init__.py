"""ARM Q4A8 model integration backed by ordinary Triton code generation."""

from .linear import (
    enable_vllm_q4_codegen,
    linear_w4a8,
    linear_w4a8_add_rmsnorm,
    linear_w4a8_rmsnorm,
    linear_w4a8_rmsnorm_qk_norm,
    pack_rhs_qsi4c32p,
    pack_rhs_qsi8cxp,
    prepare_w8_weight_kai,
    prepare_weight_asym,
    prepare_weight,
    quantize_q4_0,
    stats,
)
from .aten_linear import (
    AtenQ4Linear,
    prepare_aten_grouped_weight,
    prepare_aten_weight,
)
from .optimize_qwen3 import (
    Q4Linear,
    TritonEmbedding,
    optimize_qwen3_q4,
    optimize_qwen3_q4_aten,
    set_fused_input_rmsnorm_qkv_enabled,
    set_fused_post_add_rmsnorm_gateup_enabled,
    set_fused_qk_norm_qkv_enabled,
    set_decode_attention_fastpath_enabled,
)

enable_vllm_quant_codegen = enable_vllm_q4_codegen

__all__ = [
    "enable_vllm_q4_codegen",
    "enable_vllm_quant_codegen",
    "linear_w4a8",
    "linear_w4a8_add_rmsnorm",
    "linear_w4a8_rmsnorm",
    "linear_w4a8_rmsnorm_qk_norm",
    "pack_rhs_qsi4c32p",
    "pack_rhs_qsi8cxp",
    "prepare_w8_weight_kai",
    "prepare_weight",
    "prepare_weight_asym",
    "quantize_q4_0",
    "stats",
    "Q4Linear",
    "AtenQ4Linear",
    "TritonEmbedding",
    "prepare_aten_weight",
    "prepare_aten_grouped_weight",
    "optimize_qwen3_q4",
    "optimize_qwen3_q4_aten",
    "set_fused_input_rmsnorm_qkv_enabled",
    "set_fused_post_add_rmsnorm_gateup_enabled",
    "set_fused_qk_norm_qkv_enabled",
    "set_decode_attention_fastpath_enabled",
]
