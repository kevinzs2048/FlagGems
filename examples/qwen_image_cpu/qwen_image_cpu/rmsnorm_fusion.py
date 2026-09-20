"""Opt-in, exact-rounding Q/K RMSNorm for the current Arm CPU runtime."""

from functools import lru_cache
from types import MethodType

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("rmsnorm_fusion")


def enable_rmsnorm(module):
    from .reference import transformer_source

    native = ops()
    count = 0
    for attention in module.modules():
        if not isinstance(attention, transformer_source.QwenImage21Attention):
            continue
        for layer in (attention.norm_q, attention.norm_k):
            if hasattr(layer, "_fused_rmsnorm_original"):
                continue
            if (
                layer.weight is None
                or tuple(layer.weight.shape) != (128,)
                or layer.bias is not None
            ):
                continue
            layer._fused_rmsnorm_original = layer.forward
            layer._fused_rmsnorm_enabled = True

            def forward(self, x):
                if (
                    self._fused_rmsnorm_enabled
                    and not torch.is_grad_enabled()
                    and x.device.type == "cpu"
                    and x.dtype == torch.bfloat16
                    and x.ndim > 0
                    and x.shape[-1] == 128
                    and self.weight.device.type == "cpu"
                    and self.weight.dtype == torch.bfloat16
                    and self.bias is None
                ):
                    return native.rmsnorm128(x, self.weight, self.eps)
                return self._fused_rmsnorm_original(x)

            layer.forward = MethodType(forward, layer)
            count += 1
    return count
