"""Reuse Qwen-Image-2.1's shared modulation within one Transformer forward.

No cache crosses a timestep/request. Operations and BF16 rounding are unchanged.
Original reference classes remain unmodified on disk.
"""

from contextvars import ContextVar
from types import MethodType

import torch


def _residual(block, x, branch, gate):
    if (
        getattr(block, "_fused_residual_enabled", False)
        and x.device.type == "cpu"
        and x.ndim == 3
        and x.dtype == torch.bfloat16
        and branch.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
    ):
        return block._fused_residual_op(x, branch, gate)
    return x + gate * branch


def _norm_scale(block, norm, x, scale):
    if (
        getattr(block, "_fused_normscale_enabled", False)
        and x.device.type == "cpu"
        and x.dtype == torch.bfloat16
        and scale.dtype == torch.bfloat16
        and x.ndim == 3
    ):
        return block._fused_normscale_op(x, scale, norm.eps)
    return norm(x) * scale


def enable_shared_modulation(model):
    if getattr(model, "_shared_modulation_enabled", False):
        return 0
    from . import reference

    source = reference.transformer_source
    context = ContextVar("qwen21_modulation", default=None)
    blocks = list(model.transformer_blocks)
    if not blocks or not all(
        isinstance(b, source.QwenImage21TransformerBlock) for b in blocks
    ):
        raise ValueError(
            "Shared modulation adapter requires the matching QwenImage21 reference blocks"
        )
    for block in blocks:
        block._shared_modulation_original = block.forward

        def forward(
            self,
            hidden_states,
            modulation,
            rotary_emb=None,
            attention_mask=None,
            target_token_mask=None,
            layer_cache=None,
            kv_cache_mode=None,
            cache_write_slice=None,
            segments=None,
            key_valid=None,
        ):
            cache = context.get()
            kwargs = dict(
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
                target_token_mask=target_token_mask,
                layer_cache=layer_cache,
                kv_cache_mode=kv_cache_mode,
                cache_write_slice=cache_write_slice,
                segments=segments,
                key_valid=key_valid,
            )
            if cache is None:
                return self._shared_modulation_original(
                    hidden_states, modulation, **kwargs
                )
            key = (id(modulation), id(target_token_mask))
            if key not in cache:
                scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
                select = source._select_modulation_rows
                values = (
                    1 + select(scale1, target_token_mask),
                    select(gate1, target_token_mask).tanh(),
                    1 + select(scale2, target_token_mask),
                    select(gate2, target_token_mask).tanh(),
                )
                # Strong references prevent id reuse during this forward.
                cache[key] = (modulation, target_token_mask, values)
            scale1, gate1, scale2, gate2 = cache[key][2]
            attn_output = self.attn(
                hidden_states=_norm_scale(self, self.img_norm1, hidden_states, scale1),
                attention_mask=attention_mask,
                rotary_emb=rotary_emb,
                layer_cache=layer_cache,
                kv_cache_mode=kv_cache_mode,
                cache_write_slice=cache_write_slice,
                segments=segments,
                key_valid=key_valid,
            )
            hidden_states = _residual(self, hidden_states, attn_output, gate1)
            hidden_states = _residual(
                self,
                hidden_states,
                self.img_mlp(_norm_scale(self, self.img_norm2, hidden_states, scale2)),
                gate2,
            )
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            return hidden_states

        block.forward = MethodType(forward, block)
    original = model.forward

    def forward(self, *args, **kwargs):
        enabled = not self.training and not torch.is_grad_enabled()
        token = context.set({} if enabled else None)
        try:
            return original(*args, **kwargs)
        finally:
            context.reset(token)

    model.forward = MethodType(forward, model)
    model._shared_modulation_enabled = True
    return len(blocks)
