import json

import pytest
import torch
from qwen_image_cpu.export_w8 import export_checkpoint
from qwen_image_cpu.w8a8_sme import load_transformer_w8, selected_layer
from safetensors import safe_open
from safetensors.torch import save_file


class Tiny32(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList()
        for _ in range(32):
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.img_mlp = torch.nn.Module()
            for name in ("to_q", "to_k", "to_v"):
                setattr(block.attn, name, torch.nn.Linear(16, 8, bias=False))
            for name in ("proj", "gate_layer"):
                setattr(block.img_mlp, name, torch.nn.Linear(16, 8, bias=False))
            self.transformer_blocks.append(block)

    @classmethod
    def from_config(cls, config):
        return cls()


def source_fixture(path):
    torch.manual_seed(92)
    tensors = Tiny32().bfloat16().state_dict()
    root = path / "transformer"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"num_layers": 32}))
    save_file(tensors, root / "weights.safetensors")
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
    )
    (path / "model_index.json").write_text("{}")
    for name in ("text_encoder", "vae", "processor", "scheduler"):
        (path / name).mkdir()
        (path / name / "config.json").write_text("{}")
    save_file({"weight": torch.randn(4, 4)}, path / "vae/model.safetensors")
    return tensors


def test_export_reload_and_unquantized_components(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    original = source_fixture(source)
    report = export_checkpoint(source, output, shard_bytes=2048)
    model, metadata = load_transformer_w8(output, Tiny32)
    assert metadata["linear_count"] == report["quantized_linears"] == 112
    state = model.state_dict()
    for key, weight in original.items():
        layer = key.removesuffix(".weight")
        if selected_layer(layer, "same112"):
            scale = weight.float().abs().amax(-1) / 127
            expected = (
                (weight.float() / scale[:, None])
                .round()
                .clamp(-127, 127)
                .to(torch.int8)
            )
            assert torch.equal(state[layer + ".weight_codes"], expected)
            assert torch.equal(state[layer + ".weight_scale"], scale)
        else:
            assert torch.equal(state[key], weight)
    a, b = source / "vae/model.safetensors", output / "vae/model.safetensors"
    assert a.read_bytes() == b.read_bytes() and not b.is_symlink()
    with safe_open(b, framework="pt") as f:
        assert f.get_tensor("weight").dtype == torch.float32
    assert (output / "EXPORT_COMPLETE").exists()
    with pytest.raises(FileExistsError):
        export_checkpoint(source, output)


@pytest.mark.parametrize("failure", ["index", "nonfinite", "wrong_count"])
def test_failed_export_is_not_loadable(tmp_path, failure):
    source, output = tmp_path / "source", tmp_path / "output"
    tensors = source_fixture(source)
    root = source / "transformer"
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if failure == "index":
        index["weight_map"].pop(next(iter(tensors)))
    elif failure == "nonfinite":
        tensors[next(iter(tensors))][0, 0] = float("nan")
    else:
        key = "transformer_blocks.2.attn.to_q.weight"
        del tensors[key]
        del index["weight_map"][key]
    index_path.write_text(json.dumps(index))
    save_file(tensors, root / "weights.safetensors")
    with pytest.raises(ValueError):
        export_checkpoint(source, output)
    assert not (output / "EXPORT_COMPLETE").exists()
    assert (output / "EXPORT_FAILED").exists()
