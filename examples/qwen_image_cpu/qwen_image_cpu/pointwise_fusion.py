"""Exact BF16 SiLU table and fused rounded-BF16 SiLU-times-up inference."""

from functools import lru_cache
from types import MethodType

import torch
from torch.nn import functional as F


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("pointwise_fusion")


@lru_cache(None)
def silu_table():
    with torch.inference_mode():
        encodings = (
            torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        )
        return F.silu(encodings).contiguous()


def enable_swiglu(module, *, packed=False):
    from .reference import transformer_source

    native, table = ops(), silu_table()
    from .bf16_sme import prepare_packed_weight
    from .bf16_variants import ops as variant_ops

    matrix = variant_ops() if packed else None
    count = 0
    for layer in module.modules():
        if not isinstance(layer, transformer_source.QwenImage21SwiGLUFeedForward):
            continue
        if hasattr(layer, "_fused_swiglu_original"):
            continue
        layer._fused_swiglu_original = layer.forward
        layer._fused_swiglu_enabled = True
        layer._fused_swiglu_packed = packed

        def forward(self, x):
            if (
                not self._fused_swiglu_enabled
                or x.device.type != "cpu"
                or x.dtype != torch.bfloat16
                or torch.is_grad_enabled()
            ):
                return self._fused_swiglu_original(x)
            gate = self.gate_layer(x)
            up = self.proj(x)
            if (
                self._fused_swiglu_packed
                and matrix is not None
                and getattr(self.out, "_bf16_sme_enabled", False)
                and not self.out._bf16_sme_kernel_owned
                and gate.numel() // gate.shape[-1] >= 128
            ):
                k = gate.shape[-1]
                lhs = native.swiglu_pack(
                    gate.reshape(-1, k), up.reshape(-1, k), table, matrix.packed_rows()
                )
                rhs = prepare_packed_weight(self.out)
                workers = min(self.out._bf16_sme_workers, torch.get_num_threads())
                if (
                    getattr(self.out, "_bf16_register_epilogue_enabled", False)
                    and self.out.bias is None
                ):
                    from .bf16_register_epilogue import ops as register_ops

                    out = register_ops().matmul(
                        lhs,
                        rhs,
                        gate.numel() // k,
                        self.out.out_features,
                        k,
                        workers,
                        32,
                        256,
                    )
                else:
                    out = matrix.matmul(
                        lhs,
                        rhs,
                        gate.numel() // k,
                        self.out.out_features,
                        k,
                        workers,
                        32,
                        256,
                        2,
                        0,
                    )
                if self.out.bias is not None:
                    out = out + self.out.bias.float()
                return out.reshape(*gate.shape[:-1], self.out.out_features).to(
                    gate.dtype
                )
            return self.out(native.swiglu(gate, up, table))

        layer.forward = MethodType(forward, layer)
        count += 1
    return count


def enable_residual(module):
    if not getattr(module, "_shared_modulation_enabled", False):
        raise ValueError("Fused residual requires the shared modulation adapter")
    native = ops()
    count = 0
    for block in module.transformer_blocks:
        if getattr(block, "_fused_residual_enabled", False):
            continue
        block._fused_residual_enabled = True
        block._fused_residual_op = native.residual
        count += 1
    return count
