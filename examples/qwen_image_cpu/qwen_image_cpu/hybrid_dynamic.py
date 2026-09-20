"""Opt-in W8 SME/NEON dynamic tiling; frozen same-code A8/W8 arithmetic."""

import subprocess
from types import MethodType

import torch

from .w8a8_sme import DynamicW8SMELinear, ops


def prepare_weight(layer):
    def version(tensor):
        return None if tensor.is_inference() else tensor._version

    key = (
        id(layer.weight_codes),
        layer.weight_codes.data_ptr(),
        version(layer.weight_codes),
        id(layer.weight_scale),
        layer.weight_scale.data_ptr(),
        version(layer.weight_scale),
        tuple(layer.weight_codes.shape),
        layer.weight_codes.dtype,
        layer.weight_scale.dtype,
    )
    if getattr(layer, "_dynamic_hybrid_weight_key", None) != key:
        layer._dynamic_hybrid_weights = {}
        layer._dynamic_hybrid_weight_key = key
    fraction = layer._dynamic_hybrid_policy[2]
    n = layer.out_features
    split = int(n * fraction) // 64 * 64
    if split not in layer._dynamic_hybrid_weights:
        if split == 0 or n - split <= 0:
            raise ValueError("Dynamic hybrid requires N >= 128")
        native = ops()
        layer._dynamic_hybrid_weights[split] = (
            split,
            native.pack(layer.weight_codes[:split], layer.weight_scale[:split]),
            native.pack_neon(layer.weight_codes[split:], layer.weight_scale[split:]),
        )
    return layer._dynamic_hybrid_weights[split]


def enable_dynamic_hybrid(module):
    if torch.get_num_threads() < 18:
        raise ValueError("Dynamic hybrid policy requires at least 18 Torch threads")
    if (
        subprocess.check_output(
            ["sysctl", "-n", "hw.optional.arm.FEAT_I8MM"], text=True
        ).strip()
        != "1"
    ):
        raise RuntimeError("Dynamic hybrid requires NEON I8MM")
    native = ops()
    count = 0
    for layer in module.modules():
        if not isinstance(layer, DynamicW8SMELinear):
            continue
        if hasattr(layer, "_dynamic_hybrid_original"):
            continue
        layer._dynamic_hybrid_original = layer.forward
        layer._dynamic_hybrid_enabled = True
        layer._dynamic_hybrid_policy = (3, 15, 0.5)
        layer._dynamic_hybrid_trace_enabled = False
        layer._dynamic_hybrid_cooperative = False
        layer._dynamic_hybrid_neon_mask = None

        def forward(self, inputs):
            if (
                not self._dynamic_hybrid_enabled
                or torch.is_grad_enabled()
                or not self.activation_quantization
                or self.kernel_owned
                or self.hybrid
                or inputs.device.type != "cpu"
                or inputs.dtype != torch.bfloat16
                or inputs.numel() // inputs.shape[-1] < 128
                or self.out_features < 128
            ):
                return self._dynamic_hybrid_original(inputs)
            if torch.get_num_threads() < 18:
                raise ValueError("Dynamic hybrid requires 18 available threads")
            x = inputs.reshape(-1, self.in_features)
            split, ps, pn = prepare_weight(self)
            nmt, nnt = (128, 128) if self.out_features >= 8192 else (32, 64)
            sw, nw, _ = self._dynamic_hybrid_policy
            if self._dynamic_hybrid_cooperative:
                arguments = (
                    x,
                    ps,
                    pn,
                    split,
                    self.out_features - split,
                    sw + nw,
                    self._dynamic_hybrid_clusters,
                    64,
                    256,
                    nmt,
                    nnt,
                )
                execute = (
                    native.linear_hybrid_cooperative_trace
                    if self._dynamic_hybrid_trace_enabled
                    else native.linear_hybrid_cooperative
                )
                if self._dynamic_hybrid_neon_mask is not None:
                    arguments = (
                        arguments[:7]
                        + (self._dynamic_hybrid_neon_mask,)
                        + arguments[7:]
                    )
                    execute = (
                        native.linear_hybrid_policy_trace
                        if self._dynamic_hybrid_trace_enabled
                        else native.linear_hybrid_policy
                    )
            else:
                arguments = (
                    x,
                    ps,
                    pn,
                    split,
                    self.out_features - split,
                    sw,
                    nw,
                    64,
                    256,
                    nmt,
                    nnt,
                )
                execute = (
                    native.linear_hybrid_dynamic_trace
                    if self._dynamic_hybrid_trace_enabled
                    else native.linear_hybrid_dynamic
                )
            if self._dynamic_hybrid_trace_enabled:
                out, self._dynamic_hybrid_last_trace = execute(*arguments)
            else:
                out = execute(*arguments)
            if self.bias is not None:
                out = out + self.bias.float()
            return out.reshape(*inputs.shape[:-1], self.out_features).to(inputs.dtype)

        layer.forward = MethodType(forward, layer)
        count += 1
    return count


def configure_dynamic_hybrid(module, sme_workers, neon_workers, sme_fraction):
    if (
        not isinstance(sme_workers, int)
        or not isinstance(neon_workers, int)
        or min(sme_workers, neon_workers) < 1
        or sme_workers + neon_workers > 18
        or not 0 < sme_fraction < 1
    ):
        raise ValueError(
            "Expected positive worker counts totaling at most 18 and a strict fraction"
        )
    for layer in module.modules():
        if hasattr(layer, "_dynamic_hybrid_policy"):
            layer._dynamic_hybrid_policy = (
                sme_workers,
                neon_workers,
                float(sme_fraction),
            )
            layer._dynamic_hybrid_cooperative = False
            layer._dynamic_hybrid_neon_mask = None


def configure_cooperative_hybrid(module, workers, sme_fraction, cpu_clusters):
    if (
        not isinstance(workers, int)
        or not 2 <= workers <= 18
        or not 0 < sme_fraction < 1
        or cpu_clusters.device.type != "cpu"
        or cpu_clusters.dtype != torch.long
        or cpu_clusters.ndim != 1
        or not cpu_clusters.is_contiguous()
        or not cpu_clusters.numel()
    ):
        raise ValueError("Invalid cooperative worker policy or CPU cluster map")
    for layer in module.modules():
        if hasattr(layer, "_dynamic_hybrid_policy"):
            layer._dynamic_hybrid_policy = (1, workers - 1, float(sme_fraction))
            layer._dynamic_hybrid_cooperative = True
            layer._dynamic_hybrid_neon_mask = None
            layer._dynamic_hybrid_clusters = cpu_clusters


def configure_cluster_policy(
    module, workers, sme_fraction, cpu_clusters, neon_cluster_mask
):
    if type(neon_cluster_mask) is not int or not 0 <= neon_cluster_mask <= 0xFFFFFFFF:
        raise ValueError(
            "NEON cluster preference mask must be an unsigned 32-bit integer"
        )
    configure_cooperative_hybrid(module, workers, sme_fraction, cpu_clusters)
    for layer in module.modules():
        if hasattr(layer, "_dynamic_hybrid_policy"):
            layer._dynamic_hybrid_neon_mask = neon_cluster_mask


def select_dynamic_hybrid(module, enabled):
    for layer in module.modules():
        if hasattr(layer, "_dynamic_hybrid_enabled"):
            layer._dynamic_hybrid_enabled = enabled


def prepack_dynamic_hybrid(module):
    count = 0
    for layer in module.modules():
        if hasattr(layer, "_dynamic_hybrid_enabled"):
            prepare_weight(layer)
            count += 1
    return count
