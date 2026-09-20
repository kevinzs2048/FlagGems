"""Preserve FP32 CPU FlashAttention math while using head-major storage."""

import torch


def dispatch(query, key, value, *args, **kwargs):
    from . import reference

    if (
        torch.is_grad_enabled()
        or query.device.type != "cpu"
        or query.dtype != torch.bfloat16
        or any(
            x.ndim != 4 or x.device.type != "cpu" or x.dtype != torch.bfloat16
            for x in (key, value)
        )
        or query.ndim != 4
        or query.shape[1] < 128
    ):
        return reference._cpu_dispatch(query, key, value, *args, **kwargs)
    converted = [
        x.transpose(1, 2)
        .to(dtype=torch.float32, memory_format=torch.contiguous_format)
        .transpose(1, 2)
        for x in (query, key, value)
    ]
    output = reference._original_dispatch(*converted, *args, **kwargs)
    return output.to(dtype=query.dtype, memory_format=torch.contiguous_format)


def enable_attention_layout():
    from .reference import transformer_source

    transformer_source.dispatch_attention_fn = dispatch
