"""Patch Qwen3DecoderLayer.forward to use fused add+RMSNorm for the
post-attention layer-norm step.

The vanilla forward does:
  residual = hidden_states
  hidden_states = self.input_layernorm(hidden_states)
  hidden_states, _ = self.self_attn(...)
  hidden_states = residual + hidden_states          # ATen add (M=1×D)
  residual = hidden_states                           # alias
  hidden_states = self.post_attention_layernorm(hidden_states)  # ATen rmsnorm
  hidden_states = self.mlp(hidden_states)
  hidden_states = residual + hidden_states          # ATen add
  return hidden_states

We fuse the highlighted add + post_attention_layernorm into a single call.
This drops 1 ATen add + 1 ATen rmsnorm dispatch per layer.

Skips fusion when:
- shape is non-decode (T>1)
- dtype is not BF16
"""
import logging

import torch

from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import fused_add_rms_norm

logger = logging.getLogger(__name__)
_PATCHED: dict = {}


def _make_patched_forward(orig_forward):
    def patched_forward(self, hidden_states, **kwargs):
        # Eligibility: decode T=1, BF16 only
        if not (
            hidden_states.dim() == 3
            and hidden_states.shape[1] == 1
            and hidden_states.dtype == torch.bfloat16
        ):
            return orig_forward(self, hidden_states, **kwargs)

        residual = hidden_states
        # The ordinary-Triton Q4 coordinator can consume the raw decode row
        # and fold this RMSNorm into its already joined Q/K/V projection.
        # Other backends and prefill shapes retain the regular module call.
        qkv = getattr(self.self_attn, "_triton_qkv_coordinator", None)
        prepare_input_norm = getattr(qkv, "prepare_input_norm", None)
        if prepare_input_norm is None or not prepare_input_norm(
            hidden_states, self.input_layernorm
        ):
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            **kwargs,
        )
        # Fuse: residual = residual + hidden_states ; hidden_states = post_attn_ln(residual)
        # fused_add_rms_norm modifies both in-place: residual := residual + hidden_states,
        # hidden_states := rms_norm(residual) * weight
        can_fuse_mlp = getattr(self.mlp, "can_fuse_add_rmsnorm", None)
        if can_fuse_mlp is not None and can_fuse_mlp(
            hidden_states, residual, self.post_attention_layernorm
        ):
            finish_with_residual = getattr(
                self.mlp, "forward_add_rmsnorm_and_residual", None
            )
            if finish_with_residual is not None:
                return finish_with_residual(
                    hidden_states,
                    residual,
                    self.post_attention_layernorm,
                )
            hidden_states, residual = self.mlp.forward_add_rmsnorm(
                hidden_states,
                residual,
                self.post_attention_layernorm,
            )
        else:
            hidden_states, residual = fused_add_rms_norm(
                hidden_states.contiguous(),
                residual.contiguous(),
                normalized_shape=(
                    self.post_attention_layernorm.weight.shape[0],
                ),
                weight=self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
            )
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    return patched_forward


def patch_qwen3_layer_norm() -> int:
    """Monkey-patch Qwen3DecoderLayer.forward to use fused add+rmsnorm.

    Returns count of patched modules (qwen3 + qwen3_5).
    """
    # Targets regular Qwen3 only. Qwen3.5 has GDN/mamba layers with
    # different forward structure; not applicable here.
    targets = [
        "transformers.models.qwen3.modeling_qwen3",
    ]
    n = 0
    for modname in targets:
        try:
            mod = __import__(modname, fromlist=["Qwen3DecoderLayer"])
        except (ImportError, AttributeError):
            continue
        cls_name = (
            "Qwen3DecoderLayer" if "qwen3_5" not in modname else "Qwen3_5DecoderLayer"
        )
        if not hasattr(mod, cls_name):
            cls_name = "Qwen3DecoderLayer"
        if not hasattr(mod, cls_name):
            continue
        cls = getattr(mod, cls_name)
        key = (modname, cls_name)
        if key in _PATCHED:
            continue
        orig = cls.forward
        _PATCHED[key] = (cls, orig)
        cls.forward = _make_patched_forward(orig)
        n += 1
        logger.info(f"Patched {modname}.{cls_name}.forward")
    return n


def unpatch_qwen3_layer_norm() -> int:
    n = 0
    for key, (cls, orig) in list(_PATCHED.items()):
        cls.forward = orig
        del _PATCHED[key]
        n += 1
    return n
