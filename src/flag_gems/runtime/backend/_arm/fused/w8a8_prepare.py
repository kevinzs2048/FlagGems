# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory online W8 packing for an unquantized ARM output head."""
import torch

from .native_ops import arm_native_ops


def prepare_w8a8_weight(weight, *, chunk_rows=1024):
    """Per-channel symmetric W8; execution uses dynamic asymmetric A8.

    This is runtime weight preparation, not a checkpoint generation script.
    Only one chunk is promoted to FP32 at a time. The selected Arm library
    owns its four-channel packed ABI; standalone and KAI layouts differ.
    """
    if (weight.device.type != 'cpu' or weight.ndim != 2
            or not weight.is_floating_point()):
        raise ValueError('W8 preparation requires a floating-point CPU [N,K] weight')
    n,k = weight.shape
    if n <= 0 or k <= 0 or n % 4 or k % 32:
        raise ValueError('W8 preparation requires N%4=0 and K%32=0')
    if chunk_rows <= 0 or chunk_rows % 4:
        raise ValueError('chunk_rows must be a positive multiple of four')
    operators = arm_native_ops('w8a8')
    packed = torch.empty((n//4,4*k+48),dtype=torch.uint8)
    for begin in range(0,n,chunk_rows):
        end = min(begin+chunk_rows,n)
        values = weight[begin:end].detach().float()
        if not torch.isfinite(values).all():
            raise ValueError('W8 preparation requires finite weights')
        scales = (values.abs().amax(dim=1)/127).clamp_min_(1e-8)
        quantized = (values/scales[:,None]).round().clamp_(-127,127).to(torch.int8)
        chunk = operators.pack(quantized.contiguous(),scales.contiguous(),False)
        packed[begin//4:end//4].copy_(chunk.reshape((end-begin)//4,4*k+48))
    return packed.reshape(-1)
