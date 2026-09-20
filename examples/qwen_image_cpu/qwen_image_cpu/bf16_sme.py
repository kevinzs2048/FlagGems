from functools import lru_cache
from types import MethodType

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("bf16_sme")


def prepare_packed_weight(layer):
    """Reuse the original module cache key for adapters consuming packed LHS."""
    version = None if layer.weight.is_inference() else layer.weight._version
    weight_key = (
        layer.weight.data_ptr(),
        version,
        tuple(layer.weight.shape),
        layer.weight.dtype,
        layer.weight.device,
    )
    if layer._bf16_sme_packed is None or layer._bf16_sme_weight_key != weight_key:
        layer._bf16_sme_packed = ops().pack_direct(layer.weight)
        layer._bf16_sme_weight_key = weight_key
    return layer._bf16_sme_packed


def enable_bf16_sme(
    module, *, workers=6, axis=1, dynamic=False, kernel_owned=False, neon_pack=False
):
    count = 0
    native = ops()
    for child in module.modules():
        if not isinstance(child, torch.nn.Linear) or getattr(
            child, "_bf16_sme_enabled", False
        ):
            continue
        if child.weight.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 weight")
        child._bf16_sme_original = child.forward
        child._bf16_sme_packed = None
        child._bf16_sme_weight_key = None
        child._bf16_sme_enabled = True
        child._bf16_sme_workers = workers
        child._bf16_sme_axis = axis
        child._bf16_sme_dynamic = dynamic
        child._bf16_sme_kernel_owned = kernel_owned
        child._bf16_sme_neon_pack = neon_pack
        child._fp32_weight_cache = child._fp32_bias_cache = None

        def forward(self, x):
            if x.device.type != "cpu" or x.dtype != torch.bfloat16:
                return self._bf16_sme_original(x)
            prepare_packed_weight(self)
            flat = x.reshape(-1, self.in_features)
            workers = (
                self._bf16_sme_workers
                if self._bf16_sme_kernel_owned
                else min(self._bf16_sme_workers, torch.get_num_threads())
            )
            if (
                self._bf16_sme_dynamic
                and flat.shape[0] >= 128
                and self.out_features >= 256
            ):
                if (
                    getattr(self, "_bf16_register_epilogue_enabled", False)
                    and self.bias is None
                    and self._bf16_sme_neon_pack
                    and not self._bf16_sme_kernel_owned
                ):
                    from .bf16_register_epilogue import ops as register_ops

                    lhs = native.pack_lhs_neon(flat)
                    out = register_ops().matmul(
                        lhs,
                        self._bf16_sme_packed,
                        flat.shape[0],
                        self.out_features,
                        self.in_features,
                        workers,
                        32,
                        256,
                    )
                else:
                    linear = (
                        native.linear_dynamic_neon_pack
                        if self._bf16_sme_neon_pack
                        else native.linear_dynamic
                    )
                    out = linear(
                        flat,
                        self._bf16_sme_packed,
                        self.out_features,
                        self.in_features,
                        workers,
                        32,
                        256,
                    )
            else:
                out = native.linear(
                    flat,
                    self._bf16_sme_packed,
                    self.out_features,
                    self.in_features,
                    workers,
                    self._bf16_sme_axis,
                )
            if self.bias is not None:
                out = out + self.bias.float()
            return out.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

        child.forward = MethodType(forward, child)
        count += 1
    return count
