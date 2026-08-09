"""Monkey-patch Qwen3_5GatedDeltaNet.causal_conv1d_update with ordinary
Triton CPU codegen.

The HF fallback torch_causal_conv1d_update goes through aten::conv1d
(groups=conv_dim, kernel_size=4) which uses MKL-DNN with high dispatch
overhead (~700us/call on ARM ACL). Profile shows 7% of decode time spent
here on Qwen3.5-2B BF16.

The generated loop does state-roll, depthwise convolution and SiLU in one
launch.  It has no coarse runtime operation and no hand-written C body.

Decode (T=1, kernel_size=4, BF16) only — other shapes fall back to torch.
"""
import logging

import torch
import triton
import triton.language as tl

from ..vector_config import NONLINEAR_ROLLED_TILE

logger = logging.getLogger(__name__)

_PATCHED: set = set()
_DUMMY_BIAS = torch.zeros(1, dtype=torch.bfloat16)


@triton.jit
def _causal_conv1d_update_kernel(
    hidden_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    kernel_size: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    # Qwen3.5 fixes this depthwise filter at four taps.  Keeping the channel
    # loop rolled bounds live vectors while still exposing the four FMAs and
    # the state update to LLVM.
    for base in tl.range(0, C, BLOCK_SIZE, loop_unroll_factor=1):
        channel = base + lanes
        mask = channel < C
        # HF keeps ``kernel_size`` cached values.  After appending the new
        # token, the convolution consumes old state[1:4] plus hidden and the
        # cache becomes that same four-value window.
        state_base = (batch * C + channel) * kernel_size
        weight_base = channel * kernel_size
        s0 = tl.load(state_ptr + state_base + 1, mask=mask, other=0.0).to(tl.float32)
        s1 = tl.load(state_ptr + state_base + 2, mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(state_ptr + state_base + 3, mask=mask, other=0.0).to(tl.float32)
        hidden = tl.load(
            hidden_ptr + batch * C + channel, mask=mask, other=0.0
        ).to(tl.float32)
        w0 = tl.load(weight_ptr + weight_base, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(weight_ptr + weight_base + 1, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(weight_ptr + weight_base + 2, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(weight_ptr + weight_base + 3, mask=mask, other=0.0).to(tl.float32)
        acc = s0 * w0 + s1 * w1 + s2 * w2 + hidden * w3
        if has_bias:
            acc += tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        # torch_causal_conv1d_update materializes BF16 conv output before
        # F.silu.  Preserve that intermediate rounding explicitly.
        conv = acc.to(tl.bfloat16).to(tl.float32)
        result = conv / (1.0 + tl.exp(-conv))
        tl.store(out_ptr + batch * C + channel, result, mask=mask)

        tl.store(state_ptr + state_base, s0, mask=mask)
        tl.store(state_ptr + state_base + 1, s1, mask=mask)
        tl.store(state_ptr + state_base + 2, s2, mask=mask)
        tl.store(state_ptr + state_base + 3, hidden, mask=mask)



def _make_patched_fn(torch_causal_fn):
    def fn(hidden_states, conv_state, weight, bias=None, activation=None):
        # hidden_states: [B, C, T] bf16; weight: [C, kernel_size]; bias: None or [C]
        # conv_state:    [B, C, kernel_size] bf16 IN-OUT
        # activation: 'silu' or None.
        if (
            hidden_states.shape[-1] != 1
            or weight.shape[-1] != 4
            or hidden_states.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or conv_state.dtype != torch.bfloat16
            or conv_state.shape
            != (hidden_states.shape[0], hidden_states.shape[1], 4)
            or activation != "silu"
        ):
            return torch_causal_fn(hidden_states, conv_state, weight, bias, activation)

        B, C, _T = hidden_states.shape
        # [B, C] contiguous
        h = hidden_states.squeeze(-1).contiguous()
        w = weight.contiguous()
        # conv_state must be contiguous so the kernel can update it in place.
        if not conv_state.is_contiguous():
            conv_state_c = conv_state.contiguous()
        else:
            conv_state_c = conv_state
        out = torch.empty(B, C, dtype=torch.bfloat16)
        if bias is None:
            b_t = _DUMMY_BIAS
            has_bias = 0
        else:
            b_t = bias.contiguous()
            has_bias = 1
        _causal_conv1d_update_kernel[(B,)](
            h,
            conv_state_c,
            w,
            b_t,
            out,
            B=B,
            C=C,
            kernel_size=4,
            has_bias=has_bias,
            BLOCK_SIZE=NONLINEAR_ROLLED_TILE,
            num_warps=1,
            num_stages=1,
        )

        # If we made a contiguous copy of conv_state, write back.
        if conv_state_c.data_ptr() != conv_state.data_ptr():
            conv_state.copy_(conv_state_c)

        return out.unsqueeze(-1)

    return fn


def _get_qwen_gdn_classes() -> tuple:
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5GatedDeltaNet"),
        (
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            "Qwen3_5MoeGatedDeltaNet",
        ),
        (
            "transformers.models.qwen3_next.modeling_qwen3_next",
            "Qwen3NextGatedDeltaNet",
        ),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_conv1d(model) -> int:
    gdn_classes = _get_qwen_gdn_classes()
    if not gdn_classes:
        return 0
    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) not in _PATCHED:
            torch_fn = module.causal_conv1d_update
            module._original_causal_conv1d_update = torch_fn
            module.causal_conv1d_update = _make_patched_fn(torch_fn)
            _PATCHED.add(id(module))
            n += 1
    if n > 0:
        logger.info(
            "Patched %d GDN causal_conv1d_update with Triton codegen", n
        )
    return n


def unpatch_qwen3_5_conv1d(model) -> int:
    gdn_classes = _get_qwen_gdn_classes()
    if not gdn_classes:
        return 0
    n = 0
    for _name, module in list(model.named_modules()):
        if isinstance(module, gdn_classes) and id(module) in _PATCHED:
            if hasattr(module, "_original_causal_conv1d_update"):
                module.causal_conv1d_update = module._original_causal_conv1d_update
                del module._original_causal_conv1d_update
            _PATCHED.discard(id(module))
            n += 1
    return n
