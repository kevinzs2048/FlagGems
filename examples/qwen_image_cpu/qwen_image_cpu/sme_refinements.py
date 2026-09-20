"""Opt-in exact packing/fusion refinements; all matrix computation stays SME."""

from .layernorm_scale import enable_normscale
from .pointwise_fusion import enable_residual
from .qkv_mixed import enable_joint_qkv, prepack_joint_qkv, select_joint_qkv


def enable(
    module, norm_rope=False, w8_swiglu=False, register_output=False, mlp_workers=None
):
    if mlp_workers is not None:
        import torch

        if (
            not w8_swiglu
            or type(mlp_workers) is not int
            or not 1 <= mlp_workers <= torch.get_num_threads()
        ):
            raise ValueError(
                "MLP worker refinements require packed W8 SwiGLU and valid workers"
            )
    enable_joint_qkv(module, backend="paired_sme")
    enable_residual(module)
    enable_normscale(module)
    pairs = prepack_joint_qkv(module)
    if norm_rope:
        from .norm_rope import enable as enable_norm_rope

        enable_norm_rope(module)
    module._sme_refinements_norm_rope = norm_rope
    mlps = 0
    if w8_swiglu:
        from .w8_swiglu import enable as enable_w8_swiglu

        mlps = enable_w8_swiglu(module)
        if mlps != 28:
            raise ValueError("Expected 28 same112 W8 SwiGLU modules")
    module._sme_refinements_w8_swiglu = w8_swiglu
    module._sme_refinements_mlp_workers = mlp_workers
    if mlp_workers is not None:
        for layer in module.modules():
            if hasattr(layer, "_w8_swiglu_enabled"):
                layer.out._sme_refinements_original_workers = (
                    layer.out._bf16_sme_workers
                )
    module._sme_refinements_register_output = register_output
    register_modules = 0
    if register_output:
        from .bf16_register_epilogue import enable as enable_register

        register_modules = enable_register(module)
    select(module, True)
    return {
        "paired_qk_modules": pairs,
        "residual_blocks": len(module.transformer_blocks),
        "normscale_sites": 2 * len(module.transformer_blocks),
        "bf16_input_pack": "direct NEON BF16",
        "matrix_compute": "SME2; no NEON GEMM",
        "quantization_changed": False,
        "norm_rope": norm_rope,
        "w8_swiglu_modules": mlps,
        "bf16_register_output": register_output,
        "bf16_register_output_flagged_modules": register_modules,
        "fused_mlp_workers": mlp_workers,
        "mlp_workers_restore_on_disable": mlp_workers is not None,
    }


def select(module, enabled):
    if getattr(module, "_sme_refinements_register_output", False):
        from .bf16_register_epilogue import select as select_register

        select_register(module, enabled)
    if getattr(module, "_sme_refinements_w8_swiglu", False):
        from .w8_swiglu import select as select_w8_swiglu

        select_w8_swiglu(module, enabled)
    if getattr(module, "_sme_refinements_norm_rope", False):
        from .norm_rope import select as select_norm_rope

        select_norm_rope(module, enabled)
    select_joint_qkv(module, enabled)
    for block in module.transformer_blocks:
        block._fused_residual_enabled = enabled
        block._fused_normscale_enabled = enabled
    for layer in module.modules():
        if hasattr(layer, "_bf16_sme_neon_pack"):
            layer._bf16_sme_neon_pack = enabled
        if hasattr(layer, "_sme_refinements_original_workers"):
            layer._bf16_sme_workers = (
                module._sme_refinements_mlp_workers
                if enabled
                else layer._sme_refinements_original_workers
            )
