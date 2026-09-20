"""Reusable split weight cache for fused SME2/NEON W8 SwiGLU operators."""

import math

import torch

from .w8a8_sme import ops


@torch.no_grad()
def prepare_weights(gate, up, fraction):
    """Pack canonical W8 projections; inference tensors must remain immutable.

    Ordinary tensor updates/replacements invalidate the cache. This cache is
    runtime-only; portable model files retain canonical INT8 codes and scales.
    """
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("SME fraction must be finite and between zero and one")
    tensors = (gate.weight_codes, gate.weight_scale, up.weight_codes, up.weight_scale)
    if gate.weight_codes.shape != up.weight_codes.shape:
        raise ValueError("SwiGLU projection shapes must match")
    n = gate.weight_codes.shape[0]
    split = int(n * fraction) // 64 * 64
    if not 0 < split < n:
        raise ValueError("Both aligned weight partitions must be nonempty")

    def key(t):
        return (
            id(t),
            t.data_ptr(),
            None if t.is_inference() else t._version,
            tuple(t.shape),
            t.dtype,
            t.device,
        )

    version = tuple(key(t) for t in tensors)
    cached = getattr(gate, "_hybrid_swiglu_weights", None)
    if cached is None or cached[0] != version:
        cached = (version, {})
        gate._hybrid_swiglu_weights = cached
    if split not in cached[1]:
        w8 = ops()
        weights = [(p.weight_codes, p.weight_scale) for p in (gate, up)]
        sme = [w8.pack(q[:split], s[:split]) for q, s in weights]
        neon = [w8.pack_neon(q[split:], s[split:]) for q, s in weights]
        cached[1][split] = (split, *sme, *neon)
    return cached[1][split]
