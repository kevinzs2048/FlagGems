"""Experimental BF16 complex RoPE without intermediate float32 tensors."""

from functools import lru_cache

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("rope_fusion")


def enable_rope():
    from .reference import transformer_source

    current = transformer_source.apply_rotary_emb_qwen
    if hasattr(current, "_fused_rope_original"):
        return False
    native = ops()

    def apply(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
        if (
            apply.enabled
            and not use_real
            and x.device.type == "cpu"
            and x.dtype == torch.bfloat16
            and not torch.is_grad_enabled()
            and isinstance(freqs_cis, torch.Tensor)
            and freqs_cis.dtype == torch.complex64
            and x.ndim == 4
        ):
            # This FMA schedule matches the current CPU c10 complex multiply.
            return native.rope(x, freqs_cis, 2)
        return current(
            x, freqs_cis, use_real=use_real, use_real_unbind_dim=use_real_unbind_dim
        )

    apply.enabled = True
    apply._fused_rope_original = current
    transformer_source.apply_rotary_emb_qwen = apply
    return True
