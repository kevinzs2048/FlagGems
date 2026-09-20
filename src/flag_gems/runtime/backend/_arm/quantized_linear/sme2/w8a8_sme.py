"""Per-channel W8 / dynamic per-row A8 with KleidiAI SME2 (not the G32 TE prototype)."""

import subprocess
from functools import lru_cache

import torch


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("w8a8_sme")


@torch.no_grad()
def quantize_weight(weight):
    if (
        weight.device.type != "cpu"
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or weight.shape[1] % 8
    ):
        raise ValueError("Expected original CPU BF16 [N,K], K divisible by 8")
    codes = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(weight.shape[0], dtype=torch.float32)
    for row in range(0, weight.shape[0], 256):
        value = weight[row : row + 256].float()
        if not value.isfinite().all():
            raise ValueError("Nonfinite weight")
        maximum = value.abs().amax(-1)
        scale = torch.where(
            maximum > 0,
            (maximum / 127).clamp_min(torch.finfo(torch.float32).tiny),
            torch.ones_like(maximum),
        )
        codes[row : row + 256] = (
            (value / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        )
        scales[row : row + 256] = scale
    return codes.contiguous(), scales.contiguous()


def unpack_activation(packed, m, k):
    """Validation utility: recover exact codes/scale/zp consumed by the SME2 kernel."""
    mr, _, kr, sr = ops().layout()
    assert sr == 1 and kr == 4
    kp = (k + 31) // 32 * 32
    panels = packed.reshape(-1, mr * (kp + 8))
    codes = (
        panels[:, : mr * kp].contiguous().view(torch.int8).reshape(-1, kp // kr, mr, kr)
    )
    codes = codes.permute(0, 2, 1, 3).reshape(-1, kp)[:m, :k].contiguous()
    zero = (
        -panels[:, mr * kp : mr * (kp + 4)].contiguous().view(torch.int32).flatten()[:m]
    )
    scale = panels[:, mr * (kp + 4) :].contiguous().view(torch.float32).flatten()[:m]
    return codes, scale, zero


def activation_reference(inputs):
    """KAI A8: zero point nearest-even; scaled codes nearest, ties away from zero."""
    x = inputs.float()
    lo = x.amin(-1).clamp_max(0)
    hi = x.amax(-1).clamp_min(0)
    multiplier = torch.where(
        hi == lo, torch.ones_like(hi), torch.full_like(hi, 255) / (hi - lo)
    )
    a, b = lo * multiplier, hi * multiplier
    zero = (
        torch.where((-128 + a) + (127 + b) > 0, -128 - a, 127 - b)
        .clamp(-128, 127)
        .round()
        .int()
    )
    scaled = x * multiplier[:, None]
    rounded = scaled.sign() * torch.floor(scaled.abs() + 0.5)
    codes = (rounded + zero[:, None]).clamp(-128, 127).to(torch.int8)
    return codes, 1 / multiplier, zero


class DynamicW8SMELinear(torch.nn.Module):
    def __init__(self, codes, scales, bias=None):
        super().__init__()
        self.out_features, self.in_features = codes.shape
        self.register_buffer("weight_codes", codes)
        self.register_buffer("weight_scale", scales)
        self.bias = (
            torch.nn.Parameter(bias.detach(), requires_grad=False)
            if bias is not None
            else None
        )
        self._opaque = None
        self.axis = 0
        self.workers = None
        self.hybrid = False
        self.dynamic_tiles = False
        self.direct_a8_pack = False
        self.kernel_owned = False
        self._hybrid_packed = None
        self.activation_quantization = True

    def _apply(self, fn, recurse=True):
        self._opaque = None
        self._hybrid_packed = None
        return super()._apply(fn, recurse=recurse)

    def prepack(self):
        if self.hybrid:
            if self._hybrid_packed is None:
                if (
                    subprocess.check_output(
                        ["sysctl", "-n", "hw.optional.arm.FEAT_I8MM"], text=True
                    ).strip()
                    != "1"
                ):
                    raise RuntimeError("Hybrid W8 requires NEON I8MM")
                split = self.out_features // 128 * 64
                if split == 0 or self.out_features - split <= 0:
                    raise ValueError("Hybrid W8 requires at least 128 output channels")
                self._hybrid_packed = (
                    split,
                    ops().pack(self.weight_codes[:split], self.weight_scale[:split]),
                    ops().pack_neon(
                        self.weight_codes[split:], self.weight_scale[split:]
                    ),
                )
                self._opaque = None
            return self._hybrid_packed
        if self._opaque is None:
            self._opaque = ops().pack(self.weight_codes, self.weight_scale)
        return self._opaque

    def forward(self, inputs):
        if (
            inputs.device.type != "cpu"
            or inputs.dtype not in (torch.float32, torch.bfloat16)
            or inputs.shape[-1] != self.in_features
        ):
            raise ValueError("Expected CPU BF16/FP32 activations with matching K")
        x = inputs.reshape(-1, self.in_features)
        if self.activation_quantization:
            workers = (
                torch.get_num_threads()
                if self.workers is None
                else (
                    self.workers
                    if self.kernel_owned
                    else min(self.workers, torch.get_num_threads())
                )
            )
            if self.hybrid:
                if torch.get_num_threads() < 18:
                    raise ValueError(
                        "Measured hybrid W8 policy requires at least 18 CPU threads"
                    )
                split, ps, pn = self.prepack()
                value = ops().linear_hybrid(
                    x, ps, pn, split, self.out_features - split, 6, 12
                )
            elif self.direct_a8_pack and x.dtype == torch.bfloat16:
                value = ops().linear_bf16(
                    x,
                    self.prepack(),
                    self.out_features,
                    workers,
                    self.axis,
                    self.dynamic_tiles,
                )
            elif self.dynamic_tiles and x.shape[0] >= 128 and self.out_features >= 256:
                value = ops().linear_tiled(
                    x,
                    self.prepack(),
                    self.out_features,
                    workers,
                    32 if self.in_features > 4096 else 64,
                    256,
                    True,
                )
            else:
                value = ops().linear(
                    x, self.prepack(), self.out_features, workers, self.axis
                )
        else:
            # Accuracy-only W8A16 reference, deliberately not a speed benchmark.
            w = self.weight_codes.float() * self.weight_scale[:, None]
            value = torch.nn.functional.linear(x.float(), w)
        if self.bias is not None:
            value = value + self.bias.float()
        return value.reshape(*inputs.shape[:-1], self.out_features).to(inputs.dtype)
