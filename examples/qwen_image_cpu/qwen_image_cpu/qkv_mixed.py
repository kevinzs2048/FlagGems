"""Experimental joint scheduling of W8 Q/K and retained BF16 V projections."""

from contextvars import ContextVar
from functools import lru_cache
from types import MethodType

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("qkv_mixed")


_projection_outputs = ContextVar("qwen21_joint_qkv_outputs", default=None)


def _neon_weight(layer):
    from .w8a8_sme import ops as w8_ops

    def version(t):
        return None if t.is_inference() else t._version

    key = (
        id(layer.weight_codes),
        layer.weight_codes.data_ptr(),
        version(layer.weight_codes),
        id(layer.weight_scale),
        layer.weight_scale.data_ptr(),
        version(layer.weight_scale),
    )
    if getattr(layer, "_joint_neon_key", None) != key:
        layer._joint_neon_weight = w8_ops().pack_neon(
            layer.weight_codes, layer.weight_scale
        )
        layer._joint_neon_key = key
    return layer._joint_neon_weight


def _compute(attn, x):
    from .bf16_sme import ops as bf16_ops
    from .bf16_sme import prepare_packed_weight
    from .w8a8_sme import ops as w8_ops

    m = x.numel() // x.shape[-1]
    k = x.shape[-1]
    n = attn.to_q.out_features
    flat = x.reshape(m, k)
    bf16, w8 = bf16_ops(), w8_ops()
    workers = min(18, torch.get_num_threads())
    if attn._joint_qkv_backend == "paired_sme":
        raw = w8.linear_pair_bf16(
            flat, attn.to_q.prepack(), attn.to_k.prepack(), n, workers
        )
        result = []
        for out, layer in zip(raw, [attn.to_q, attn.to_k]):
            if layer.bias is not None:
                out = out + layer.bias.float()
            result.append(out.reshape(*x.shape[:-1], n).to(x.dtype))
        result.append(attn.to_v._joint_qkv_original(x))
        return result
    lhs = bf16.pack_lhs_neon(flat)
    a8 = w8.repack_neon_lhs(w8.quant_pack_bf16(flat), m, k)
    sw, smt, snt, nmt, nnt = attn._joint_qkv_policy
    raw = ops().matmul(
        lhs,
        a8,
        _neon_weight(attn.to_q),
        _neon_weight(attn.to_k),
        prepare_packed_weight(attn.to_v),
        m,
        n,
        k,
        workers,
        min(sw, workers - 1),
        smt,
        snt,
        nmt,
        nnt,
    )
    result = []
    for out, layer in zip(raw, [attn.to_q, attn.to_k, attn.to_v]):
        if layer.bias is not None:
            out = out + layer.bias.float()
        result.append(out.reshape(*x.shape[:-1], n).to(x.dtype))
    return result


def enable_joint_qkv(module, backend="mixed"):
    """Reuse reference norm/RoPE/cache code with scoped precomputed projections."""
    from .reference import transformer_source as source
    from .w8a8_sme import DynamicW8SMELinear

    if backend not in ("mixed", "paired_sme"):
        raise ValueError("Unknown QKV backend")
    original = source._qwenimage21_prepare_qkv
    if not getattr(original, "_joint_qkv_adapter", False):

        def prepare(attn, hidden_states, *args, **kwargs):
            eligible = (
                getattr(attn, "_joint_qkv_enabled", False)
                and not attn.training
                and not torch.is_grad_enabled()
                and attn.to_q.activation_quantization
                and attn.to_k.activation_quantization
                and not attn.to_q.hybrid
                and not attn.to_k.hybrid
                and not attn.to_q.kernel_owned
                and not attn.to_k.kernel_owned
                and attn.to_v.weight.dtype == torch.bfloat16
                and hidden_states.device.type == "cpu"
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.numel() // hidden_states.shape[-1] >= 128
                and torch.get_num_threads() >= 2
            )
            if not eligible:
                return original(attn, hidden_states, *args, **kwargs)
            values = _compute(attn, hidden_states)
            outputs = dict(zip([attn.to_q, attn.to_k, attn.to_v], values))
            token = _projection_outputs.set((hidden_states, outputs))
            try:
                return original(attn, hidden_states, *args, **kwargs)
            finally:
                _projection_outputs.reset(token)

        prepare._joint_qkv_adapter = True
        source._qwenimage21_prepare_qkv = prepare
    count = 0
    for attn in module.modules():
        if not isinstance(attn, source.QwenImage21Attention):
            continue
        if (
            not isinstance(attn.to_q, DynamicW8SMELinear)
            or not isinstance(attn.to_k, DynamicW8SMELinear)
            or not getattr(attn.to_v, "_bf16_sme_enabled", False)
        ):
            continue
        if hasattr(attn, "_joint_qkv_enabled"):
            if attn._joint_qkv_backend != backend:
                raise ValueError("QKV adapter already enabled with a different backend")
            continue
        if (
            len(
                {
                    (layer.in_features, layer.out_features)
                    for layer in [attn.to_q, attn.to_k, attn.to_v]
                }
            )
            != 1
        ):
            raise ValueError("Joint QKV requires matching projection shapes")
        attn._joint_qkv_enabled = True
        attn._joint_qkv_policy = (3, 32, 256, 32, 64)
        attn._joint_qkv_backend = backend
        for layer in [attn.to_q, attn.to_k, attn.to_v]:
            layer._joint_qkv_original = layer.forward

            def forward(self, x):
                context = _projection_outputs.get()
                if context is not None and context[0] is x and self in context[1]:
                    return context[1][self]
                return self._joint_qkv_original(x)

            layer.forward = MethodType(forward, layer)
        count += 1
    return count


def prepack_joint_qkv(module):
    from .bf16_sme import prepare_packed_weight

    count = 0
    for attn in module.modules():
        if hasattr(attn, "_joint_qkv_enabled"):
            if attn._joint_qkv_backend == "mixed":
                _neon_weight(attn.to_q)
                _neon_weight(attn.to_k)
            else:
                attn.to_q.prepack()
                attn.to_k.prepack()
            prepare_packed_weight(attn.to_v)
            count += 1
    return count


def select_joint_qkv(module, enabled):
    for attn in module.modules():
        if hasattr(attn, "_joint_qkv_enabled"):
            attn._joint_qkv_enabled = enabled
