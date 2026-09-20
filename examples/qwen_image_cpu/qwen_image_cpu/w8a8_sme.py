"""Per-channel W8 / dynamic per-row A8 with KleidiAI SME2 (not the G32 TE prototype)."""

import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

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


def selected_layer(name, policy):
    if not name.startswith("transformer_blocks."):
        return False
    block = int(name.split(".")[1])
    if policy == "same112":
        return 2 <= block < 30 and name.endswith(
            (".attn.to_q", ".attn.to_k", ".img_mlp.proj", ".img_mlp.gate_layer")
        )
    if policy == "middle196":
        return 2 <= block < 30
    if policy == "all224":
        return True
    raise ValueError("Unknown W8 policy")


@torch.no_grad()
def enable_transformer_w8a8(model, policy="same112"):
    started = time.perf_counter()
    records = []
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, torch.nn.Linear) or not selected_layer(name, policy):
            continue
        codes, scales = quantize_weight(layer.weight)
        parent, _, key = name.rpartition(".")
        setattr(
            model.get_submodule(parent),
            key,
            DynamicW8SMELinear(codes, scales, layer.bias),
        )
        records.append({"name": name, "n": layer.out_features, "k": layer.in_features})
    if len(records) != {"same112": 112, "middle196": 196, "all224": 224}[policy]:
        raise ValueError("Unexpected Transformer W8 Linear count")
    return {
        "policy": policy,
        "linear_count": len(records),
        "seconds": time.perf_counter() - started,
        "weight": "symmetric INT8 [-127,127], per-output-channel, FP32 scale",
        "activation": "dynamic asymmetric INT8 per-token; KAI rounding; FP32 scale",
        "backend": "KleidiAI SME2 INT8 MOPA, INT32 accumulation, FP32 scaling, BF16 output",
        "source": "original BF16; no W4 requantization",
        "disk_export": False,
        "layers": records,
    }


def load_transformer_w8(directory, model_class):
    from accelerate import init_empty_weights
    from safetensors import safe_open

    directory = Path(directory)
    if (
        not (directory / "EXPORT_COMPLETE").is_file()
        or (directory / "EXPORT_FAILED").exists()
    ):
        raise ValueError("Incomplete W8 export")
    root = directory / "transformer"
    quant = json.loads((root / "quantization_config.json").read_text())
    required = {
        "format_version": 1,
        "scheme": "w8a8_dynamic_per_channel",
        "weight_dtype": "int8",
        "weight_mapping": "symmetric",
        "weight_qmin": -127,
        "weight_qmax": 127,
        "weight_zero_point": 0,
        "weight_granularity": "per_output_channel",
        "weight_scale_dtype": "float32",
        "weight_rounding": "nearest_even",
        "activation_dtype": "int8",
        "activation_mapping": "asymmetric",
        "activation_granularity": "per_token",
        "activation_scale_dtype": "float32",
        "activation_code_rounding": "nearest_ties_away_from_zero",
        "activation_zero_point_rounding": "nearest_even",
        "activation_quantizer": "KleidiAI 1.29.0 kai_lhs_quant_pack_qai8dxp_f32",
        "output_dtype": "bfloat16",
        "accumulator_dtype": "int32",
        "dequant_dtype": "float32",
    }
    if any(quant.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported W8 quantization metadata")
    with init_empty_weights():
        model = model_class.from_config(json.loads((root / "config.json").read_text()))
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    for name, layer in quant["quantized_layers"].items():
        original = model.get_submodule(name)
        n, k = layer["n"], layer["k"]
        if not isinstance(original, torch.nn.Linear) or (
            original.out_features,
            original.in_features,
        ) != (n, k):
            raise ValueError("W8 model shape mismatch: " + name)
        if (
            n <= 0
            or k <= 0
            or k % 8
            or k > 32768
            or layer["weight_codes"] != name + ".weight_codes"
            or layer["weight_scale"] != name + ".weight_scale"
        ):
            raise ValueError("Invalid W8 layer metadata")
        replacement = DynamicW8SMELinear(
            torch.empty(n, k, dtype=torch.int8, device="meta"),
            torch.empty(n, dtype=torch.float32, device="meta"),
            original.bias,
        )
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, replacement)
    index = json.loads(
        (root / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    expected = model.state_dict()
    seen = set()
    if set(index) != set(expected):
        raise ValueError("W8 checkpoint state keys differ from model")
    for filename in sorted(set(index.values())):
        if Path(filename).name != filename:
            raise ValueError("Invalid shard path")
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for key, tensor in tensors.items():
            if key in seen or key not in expected or index.get(key) != filename:
                raise ValueError("Duplicate/misindexed W8 tensor")
            if (
                tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype
            ):
                raise ValueError("W8 tensor shape/dtype mismatch: " + key)
            if key.endswith(".weight_scale") and (
                not tensor.isfinite().all() or not (tensor > 0).all()
            ):
                raise ValueError("Invalid W8 scale")
            if key.endswith(".weight_codes") and (tensor == -128).any():
                raise ValueError("W8 code outside declared [-127,127]")
        model.load_state_dict(tensors, strict=False, assign=True)
        seen.update(tensors)
        del tensors
    if seen != set(expected) or any(
        t.device.type == "meta" for t in model.state_dict().values()
    ):
        raise ValueError("Unloaded W8 tensors")
    report = {
        "policy": quant["policy"],
        "linear_count": len(quant["quantized_layers"]),
        "checkpoint": str(directory.resolve()),
        "disk_export": True,
        "format": required,
        "backend": "KleidiAI SME2 INT8 MOPA",
    }
    return model.eval().requires_grad_(False), report
