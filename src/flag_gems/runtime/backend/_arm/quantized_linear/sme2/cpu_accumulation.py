"""BF16-valued weights with FP32 CPU accumulation and BF16 outputs."""

from types import MethodType

import torch
from torch.nn import functional as F


def enable_fp32_accumulation(module, *, cache_weights=True):
    count = 0
    for child in module.modules():
        if not isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)) or getattr(
            child, "_fp32_accumulation_enabled", False
        ):
            continue
        child._fp32_accumulation_enabled = True
        child._fp32_cache_weights = cache_weights
        child._fp32_weight_cache = None
        child._fp32_bias_cache = None
        original = child.forward

        def measured_linear(self, x):
            if self.weight.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
                return self._original_bf16_forward(x)
            weight, bias = fp32_weights(self)
            return F.linear(x.float(), weight, bias).to(x.dtype)

        def measured_conv(self, x, cache_x=None):
            if self.weight.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
                return (
                    self._original_bf16_forward(x, cache_x)
                    if hasattr(self, "_padding")
                    else self._original_bf16_forward(x)
                )
            weight, bias = fp32_weights(self)
            if hasattr(self, "_padding"):
                if cache_x is not None or x.ndim != 5 or x.shape[2] != 1:
                    raise ValueError(
                        "Reference 2.1 causal convolution requires one frame and no cached input"
                    )
                value = F.pad(x.squeeze(2), self._padding)
                return (
                    F.conv2d(
                        value.float(),
                        weight,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                    .to(x.dtype)
                    .unsqueeze(2)
                )
            if x.ndim == 5:
                b, c, t, h, w = x.shape
                value = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            else:
                value = x
            value = F.conv2d(
                value.float(),
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            ).to(x.dtype)
            if x.ndim == 5:
                value = value.reshape(b, t, *value.shape[1:]).permute(0, 2, 1, 3, 4)
            return value

        child._original_bf16_forward = original
        child.forward = MethodType(
            measured_linear if isinstance(child, torch.nn.Linear) else measured_conv,
            child,
        )
        count += 1
    return count


def fp32_weights(module):
    weight = module._fp32_weight_cache
    if weight is None:
        weight = module.weight.float()
        bias = module.bias.float() if module.bias is not None else None
        if module._fp32_cache_weights:
            module._fp32_weight_cache, module._fp32_bias_cache = weight, bias
    else:
        bias = module._fp32_bias_cache
    return weight, bias


def clear_fp32_caches(module):
    for child in module.modules():
        if getattr(child, "_fp32_accumulation_enabled", False):
            child._fp32_weight_cache = child._fp32_bias_cache = None
