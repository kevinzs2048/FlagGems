import json

import pytest
import torch
from qwen_image_cpu.w8a8_sme import load_transformer_w8, quantize_weight
from safetensors.torch import save_file


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 8, bias=False)
        self.norm = torch.nn.LayerNorm(8)

    @classmethod
    def from_config(cls, config):
        return cls()


def checkpoint(path):
    root = path / "transformer"
    root.mkdir(parents=True)
    (path / "EXPORT_COMPLETE").write_text("test\n")
    (root / "config.json").write_text("{}")
    q = {
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
        "policy": "fixture",
        "quantized_layers": {
            "proj": {
                "n": 8,
                "k": 16,
                "weight_codes": "proj.weight_codes",
                "weight_scale": "proj.weight_scale",
            }
        },
    }
    torch.manual_seed(83)
    w = torch.randn(8, 16).bfloat16()
    codes, scales = quantize_weight(w)
    tensors = {
        "proj.weight_codes": codes,
        "proj.weight_scale": scales,
        "norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "norm.bias": torch.zeros(8, dtype=torch.bfloat16),
    }
    save_file(tensors, root / "weights.safetensors")
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
    )
    (root / "quantization_config.json").write_text(json.dumps(q))
    return q, tensors


def test_load_codes_and_scales_exactly(tmp_path):
    _, tensors = checkpoint(tmp_path)
    model, report = load_transformer_w8(tmp_path, Tiny)
    assert report["linear_count"] == 1 and report["disk_export"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, tensors[key])


@pytest.mark.parametrize(
    "failure", ["metadata", "scale", "code", "incomplete", "missing"]
)
def test_reject_invalid_format(tmp_path, failure):
    quant, tensors = checkpoint(tmp_path)
    root = tmp_path / "transformer"
    if failure == "metadata":
        quant["weight_scale_dtype"] = "bfloat16"
        (root / "quantization_config.json").write_text(json.dumps(quant))
    elif failure == "incomplete":
        (tmp_path / "EXPORT_FAILED").write_text("interrupted")
    else:
        if failure == "scale":
            tensors["proj.weight_scale"][0] = float("nan")
        elif failure == "code":
            tensors["proj.weight_codes"][0, 0] = -128
        else:
            del tensors["norm.weight"]
        save_file(tensors, root / "weights.safetensors")
    with pytest.raises(ValueError):
        load_transformer_w8(tmp_path, Tiny)
