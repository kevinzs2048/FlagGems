"""Explicit inference-only W8 provider shared by Linear, Q/K and SwiGLU pairs.

No global ATen replacement. Native remains the production default. The counter
records GEMMs that execute, including pairs which bypass module.forward.
"""

from collections import Counter
from types import MethodType

import torch

from .w8a8_sme import DynamicW8SMELinear, ops, unpack_activation


class ArmOperatorBackend:
    def __init__(self, model, mode="native", workers=18):
        if mode not in ("native", "triton", "aten"):
            raise ValueError("Unknown CPU operator backend")
        self.mode, self.workers = mode, workers
        self.layers = {
            layer: name
            for name, layer in model.named_modules()
            if isinstance(layer, DynamicW8SMELinear)
        }
        self.packed = {}
        self.mlp_column_bf16 = False
        self.counts, self.modules = Counter(), Counter()
        if list(ops().layout()) != [16, 64, 4, 1]:
            raise RuntimeError("Unvalidated native A8 ABI / SME streaming length")

    def reset(self):
        self.counts.clear()
        self.modules.clear()

    def snapshot(self):
        return {
            "backend": self.mode,
            "calls": dict(self.counts),
            "modules": dict(self.modules),
            "distinct_w8_modules": len(self.modules),
            "scope": "Executed W8 GEMMs only; BF16, attention, packing and TE/VAE remain native/ATen",
        }

    def record(self, layers):
        for layer in layers:
            self.counts[self.mode + "_w8_gemm"] += 1
            self.modules[self.layers[layer]] += 1

    def weight(self, layer):
        from .triton_w8_linear import pack_weight

        def key(t):
            return (id(t), t.data_ptr(), None if t.is_inference() else t._version)

        version = (key(layer.weight_codes), key(layer.weight_scale))
        cached = self.packed.get(layer)
        if cached is None or cached[0] != version:
            cached = (version, pack_weight(layer.weight_codes, layer.weight_scale))
            self.packed[layer] = cached
        return cached[1]

    def prepack(self):
        if self.mode == "triton":
            for layer in self.layers:
                self.weight(layer)

    def pair(self, layers, x, workers=None, *, direct_bf16=False, column_major=False):
        if (
            torch.is_grad_enabled()
            or x.dtype != torch.bfloat16
            or x.device.type != "cpu"
        ):
            raise ValueError("CPU W8 provider requires BF16 inference inputs")
        k = layers[0].in_features
        if x.shape[-1] != k or any(p.in_features != k for p in layers):
            raise ValueError("Incompatible W8 pair dimensions")
        if direct_bf16 and (
            self.mode != "triton" or any(p.bias is not None for p in layers)
        ):
            raise ValueError(
                "Direct BF16 candidate requires bias-free Triton projections"
            )
        m = x.numel() // k
        threads = min(
            self.workers if workers is None else workers, torch.get_num_threads()
        )
        packed = ops().quant_pack_bf16(x.reshape(m, k))
        self.counts["native_a8_pack"] += 1
        result = []
        for layer in layers:
            if self.mode == "triton":
                if direct_bf16:
                    from flag_gems.runtime.backend._arm.ops.w8a8_packed_sme2 import (
                        w8a8_packed_sme2_bf16,
                    )

                    w = self.weight(layer)
                    value = w8a8_packed_sme2_bf16(
                        packed,
                        w.data,
                        w.sums,
                        w.scales,
                        m,
                        w.n,
                        w.k,
                        workers=threads,
                        column_major=column_major,
                    )
                    self.counts["triton_direct_bf16_gemm"] += 1
                else:
                    from .triton_w8_linear import from_packed_activation

                    value = from_packed_activation(
                        packed,
                        self.weight(layer),
                        m,
                        workers=threads,
                        chunk=16,
                        tile_n=16,
                        row_epilogue=True,
                    )
            elif self.mode == "native":
                value = ops().matmul(
                    packed, layer.prepack(), m, layer.out_features, k, threads, 0
                )
            else:
                codes, scale, zero = unpack_activation(packed, m, k)
                integer = torch._int_mm(
                    codes.contiguous(), layer.weight_codes.T.contiguous()
                )
                # unpack_activation returns the mathematical zero point, not -zp.
                integer = (
                    integer
                    - zero[:, None]
                    * layer.weight_codes.sum(1, dtype=torch.int32)[None, :]
                )
                value = integer.float() * (scale[:, None] * layer.weight_scale[None, :])
            if layer.bias is not None:
                value = value + layer.bias.float()
            result.append(value.reshape(*x.shape[:-1], layer.out_features).bfloat16())
        self.record(layers)
        return result


def enable(model, mode="native", workers=18, expected_layers=112):
    """Install after the native tuned fusions, so pair paths are also covered."""
    from . import qkv_mixed

    if hasattr(model, "_arm_operator_backend"):
        raise ValueError("CPU provider is already installed")
    route = ArmOperatorBackend(model, mode, workers)
    if len(route.layers) != expected_layers:
        raise ValueError("CPU provider coverage differs from frozen quantization")
    model._arm_operator_backend = route
    for layer in route.layers:
        original = layer.forward

        def forward(self, x, original=original):
            context = qkv_mixed._projection_outputs.get()
            if context is not None and context[0] is x and self in context[1]:
                return original(x)
            if route.mode == "native":
                out = original(x)
                route.record([self])
                return out
            return route.pair([self], x)[0]

        layer.forward = MethodType(forward, layer)
    # Preserve the native pair's sharing and reference norm/RoPE/cache semantics.
    if not getattr(qkv_mixed._compute, "_arm_provider", False):
        original_compute = qkv_mixed._compute

        def compute(attn, x):
            provider = getattr(attn, "_arm_operator_backend", None)
            if provider is None:
                return original_compute(attn, x)
            if provider.mode == "native":
                out = original_compute(attn, x)
                provider.record([attn.to_q, attn.to_k])
                return out
            return provider.pair([attn.to_q, attn.to_k], x) + [
                attn.to_v._joint_qkv_original(x)
            ]

        compute._arm_provider = True
        qkv_mixed._compute = compute
    for layer in model.modules():
        if getattr(layer, "_joint_qkv_backend", None) == "paired_sme":
            layer._arm_operator_backend = route
        if not hasattr(layer, "_w8_swiglu_enabled"):
            continue
        original = layer.forward

        def mlp(self, x, original=original):
            if route.mode == "native":
                before = route.modules[route.layers[self.gate_layer]]
                out = original(x)
                # Full512 inference takes the fused native path, which bypasses
                # both children. Otherwise their wrappers already count calls.
                if route.modules[route.layers[self.gate_layer]] == before:
                    route.record([self.gate_layer, self.proj])
                return out
            from .bf16_sme import prepare_packed_weight
            from .bf16_variants import ops as matrix_ops
            from .pointwise_fusion import ops as pointwise_ops
            from .pointwise_fusion import silu_table

            workers = min(self.out._bf16_sme_workers, torch.get_num_threads())
            direct = route.mode == "triton" and route.mlp_column_bf16
            gate, up = route.pair(
                [self.gate_layer, self.proj],
                x,
                18 if direct else workers,
                direct_bf16=direct,
                column_major=direct,
            )
            n = gate.shape[-1]
            m = gate.numel() // n
            lhs = pointwise_ops().swiglu_pack(
                gate.reshape(m, n),
                up.reshape(m, n),
                silu_table(),
                matrix_ops().packed_rows(),
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
                out = matrix_ops().matmul(
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
            return out.reshape(*x.shape[:-1], self.out.out_features).bfloat16()

        layer.forward = MethodType(mlp, layer)
    return route
