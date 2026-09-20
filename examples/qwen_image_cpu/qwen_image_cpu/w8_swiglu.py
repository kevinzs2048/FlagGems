"""Experimental W8 projection pair with exact SwiGLU-to-BF16 packed output."""

from functools import lru_cache
from types import MethodType

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("w8_swiglu")


def enable(module, mt=64, nt=256):
    from .bf16_sme import prepare_packed_weight
    from .bf16_variants import ops as matrix_ops
    from .pointwise_fusion import silu_table
    from .reference import transformer_source
    from .w8a8_sme import DynamicW8SMELinear
    from .w8a8_sme import ops as w8_ops

    w8, matrix, native = w8_ops(), matrix_ops(), ops()
    table = silu_table()
    count = 0
    for layer in module.modules():
        if (
            not isinstance(layer, transformer_source.QwenImage21SwiGLUFeedForward)
            or hasattr(layer, "_w8_swiglu_enabled")
            or not all(
                isinstance(p, DynamicW8SMELinear) and p.bias is None
                for p in [layer.gate_layer, layer.proj]
            )
            or not getattr(layer.out, "_bf16_sme_enabled", False)
        ):
            continue
        if not getattr(layer, "_fused_swiglu_packed", False):
            raise ValueError(
                "W8 SwiGLU experiment requires the existing packed SwiGLU baseline"
            )
        if (
            layer.gate_layer.in_features != layer.proj.in_features
            or layer.gate_layer.out_features != layer.proj.out_features
        ):
            raise ValueError("SwiGLU projection dimensions must match")
        layer._w8_swiglu_original = layer.forward
        layer._w8_swiglu_enabled = True
        layer._w8_swiglu_tile = (mt, nt)

        def forward(self, x):
            eligible = (
                self._w8_swiglu_enabled
                and not self.training
                and not torch.is_grad_enabled()
                and x.device.type == "cpu"
                and x.dtype == torch.bfloat16
                and x.numel() // x.shape[-1] >= 128
                and self._fused_swiglu_enabled
                and self._fused_swiglu_packed
                and not self.out._bf16_sme_kernel_owned
                and self.out.weight.dtype == torch.bfloat16
                and all(
                    p.activation_quantization
                    and not p.hybrid
                    and not p.kernel_owned
                    and p.direct_a8_pack
                    and p.dynamic_tiles
                    and p.bias is None
                    for p in [self.gate_layer, self.proj]
                )
            )
            if not eligible:
                return self._w8_swiglu_original(x)
            k = x.shape[-1]
            m = x.numel() // k
            n = self.proj.out_features
            workers = min(self.out._bf16_sme_workers, torch.get_num_threads())
            lhs = native.run(
                w8.quant_pack_bf16(x.reshape(m, k)),
                self.gate_layer.prepack(),
                self.proj.prepack(),
                table,
                m,
                n,
                k,
                workers,
                *self._w8_swiglu_tile,
            )
            if (
                getattr(self.out, "_bf16_register_epilogue_enabled", False)
                and self.out.bias is None
            ):
                from .bf16_register_epilogue import ops as register_ops

                out = register_ops().matmul(
                    lhs,
                    prepare_packed_weight(self.out),
                    m,
                    self.out.out_features,
                    n,
                    workers,
                    32,
                    256,
                )
            else:
                out = matrix.matmul(
                    lhs,
                    prepare_packed_weight(self.out),
                    m,
                    self.out.out_features,
                    n,
                    workers,
                    32,
                    256,
                    2,
                    0,
                )
            if self.out.bias is not None:
                out = out + self.out.bias.float()
            return out.reshape(*x.shape[:-1], self.out.out_features).to(x.dtype)

        layer.forward = MethodType(forward, layer)
        count += 1
    return count


def select(module, enabled):
    for layer in module.modules():
        if hasattr(layer, "_w8_swiglu_enabled"):
            layer._w8_swiglu_enabled = enabled
