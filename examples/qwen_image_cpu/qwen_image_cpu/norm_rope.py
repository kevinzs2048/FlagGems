"""Experimental exact BF16 RMSNorm + complex RoPE, without an intermediate."""

from contextvars import ContextVar
from functools import lru_cache
from types import MethodType

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("norm_rope")


_rotated = ContextVar("qwen21_fused_norm_rope", default=None)


def enable(module):
    """Keep reference QKV/cache handling; skip only RoPE already fused in norm.

    The scoped map holds strong references and is cleared on exceptions. The
    reference's BF16 ``.to(value.dtype)`` is an identity for this eligible path.
    """
    from .reference import transformer_source as source

    native = ops()
    original = source._qwenimage21_prepare_qkv
    if not getattr(original, "_norm_rope_adapter", False):

        def prepare(
            attn,
            hidden_states,
            rotary_emb,
            layer_cache,
            kv_cache_mode,
            cache_write_slice,
        ):
            eligible = (
                getattr(attn, "_norm_rope_enabled", False)
                and not attn.training
                and not torch.is_grad_enabled()
                and hidden_states.device.type == "cpu"
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.ndim == 3
                and isinstance(rotary_emb, torch.Tensor)
                and rotary_emb.device.type == "cpu"
                and rotary_emb.dtype == torch.complex64
                and rotary_emb.ndim == 2
                and tuple(rotary_emb.shape) == (hidden_states.shape[1], 64)
                and attn.to_v.weight.device.type == "cpu"
                and attn.to_v.weight.dtype == torch.bfloat16
                and all(
                    n.weight.device.type == "cpu"
                    and n.weight.dtype == torch.bfloat16
                    and tuple(n.weight.shape) == (128,)
                    and n.bias is None
                    for n in [attn.norm_q, attn.norm_k]
                )
            )
            if not eligible:
                return original(
                    attn,
                    hidden_states,
                    rotary_emb,
                    layer_cache,
                    kv_cache_mode,
                    cache_write_slice,
                )
            state = {
                "frequency": rotary_emb,
                "norms": (attn.norm_q, attn.norm_k),
                "pending": {},
            }
            token = _rotated.set(state)
            try:
                result = original(
                    attn,
                    hidden_states,
                    rotary_emb,
                    layer_cache,
                    kv_cache_mode,
                    cache_write_slice,
                )
                if state["pending"]:
                    raise RuntimeError(
                        "Reference QKV did not consume fused RoPE results"
                    )
                return result
            finally:
                _rotated.reset(token)

        prepare._norm_rope_adapter = True
        source._qwenimage21_prepare_qkv = prepare
    original_rope = source.apply_rotary_emb_qwen
    if not getattr(original_rope, "_norm_rope_adapter", False):

        def apply(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
            state = _rotated.get()
            if state is not None and id(x) in state["pending"]:
                if use_real or freqs_cis is not state["frequency"]:
                    raise RuntimeError(
                        "Unexpected reference RoPE dispatch in fused scope"
                    )
                assert state["pending"].pop(id(x)) is x
                return x
            return original_rope(
                x, freqs_cis, use_real=use_real, use_real_unbind_dim=use_real_unbind_dim
            )

        apply._norm_rope_adapter = True
        source.apply_rotary_emb_qwen = apply
    count = 0
    for attn in module.modules():
        if (
            not isinstance(attn, source.QwenImage21Attention)
            or hasattr(attn, "_norm_rope_enabled")
            or not isinstance(attn.to_v, torch.nn.Linear)
            or any(
                n.weight is None
                or tuple(n.weight.shape) != (128,)
                or n.bias is not None
                for n in [attn.norm_q, attn.norm_k]
            )
        ):
            continue
        attn._norm_rope_enabled = True
        for norm in [attn.norm_q, attn.norm_k]:
            norm._norm_rope_original = norm.forward

            def forward(self, x):
                state = _rotated.get()
                if state is None or self not in state["norms"]:
                    return self._norm_rope_original(x)
                out = native.run(x, self.weight, state["frequency"], self.eps)
                state["pending"][id(out)] = out
                return out

            norm.forward = MethodType(forward, norm)
        count += 1
    return count


def select(module, enabled):
    for attn in module.modules():
        if hasattr(attn, "_norm_rope_enabled"):
            attn._norm_rope_enabled = enabled
