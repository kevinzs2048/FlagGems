"""Stream original BF16 safetensors into a portable mixed per-channel W8 checkpoint."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .w8a8_sme import quantize_weight, selected_layer


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            result.update(data)
    return result.hexdigest()


@torch.inference_mode()
def export_checkpoint(source, output, shard_bytes=512 * 1024 * 1024):
    args = argparse.Namespace(
        source=Path(source).resolve(), output=Path(output).absolute(), policy="same112"
    )
    if shard_bytes < 1:
        raise ValueError("Positive shard size required")
    if args.output.resolve().is_relative_to(args.source):
        raise ValueError("Output must be outside the source model directory")
    if (args.source / "transformer/quantization_config.json").exists():
        raise ValueError(
            "Export requires original BF16 weights, not a quantized checkpoint"
        )
    config = json.loads((args.source / "transformer/config.json").read_text())
    if config.get("num_layers") != 32:
        raise ValueError("same112 requires the verified 32-block architecture")
    for component in [
        "model_index.json",
        "text_encoder",
        "vae",
        "processor",
        "scheduler",
    ]:
        if not (args.source / component).exists():
            raise ValueError("Missing source component: " + component)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "transformer"
    destination.mkdir()
    start = time.perf_counter()
    try:
        source = args.source / "transformer"
        index = json.loads(
            (source / "diffusion_pytorch_model.safetensors.index.json").read_text()
        )["weight_map"]
        config = json.loads((source / "config.json").read_text())
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        quant = {
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
            "policy": args.policy,
            "quantized_layers": {},
        }
        buffers = {}
        buffer_bytes = 0
        total_bytes = 0
        weight_map = {}
        shard_number = 0
        hashes = {}
        source_hashes = {}

        def flush():
            nonlocal buffers, buffer_bytes, shard_number
            if not buffers:
                return
            filename = f"diffusion_pytorch_model-{shard_number:05d}.safetensors"
            save_file(
                buffers,
                destination / filename,
                metadata={
                    "format": "pt",
                    "quantization": "w8a8_dynamic_per_channel_v1",
                },
            )
            for name in buffers:
                weight_map[name] = filename
            hashes[filename] = sha256(destination / filename)
            with safe_open(
                destination / filename, framework="pt", device="cpu"
            ) as check:
                if set(check.keys()) != set(buffers):
                    raise ValueError("Saved shard keys differ")
                for key, value in buffers.items():
                    actual = check.get_tensor(key)
                    if actual.dtype != value.dtype or not torch.equal(actual, value):
                        raise ValueError("Saved tensor differs: " + key)
            shard_number += 1
            buffers = {}
            buffer_bytes = 0

        def add(name, value):
            nonlocal buffer_bytes, total_bytes
            size = value.numel() * value.element_size()
            if buffers and buffer_bytes + size > shard_bytes:
                flush()
            buffers[name] = value.contiguous()
            buffer_bytes += size
            total_bytes += size

        seen = set()
        for filename in sorted(set(index.values())):
            if Path(filename).name != filename:
                raise ValueError("Invalid shard filename")
            before = (source / filename).stat()
            source_hashes[filename] = sha256(source / filename)
            print("Quantizing " + filename, flush=True)
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                for name in sorted(handle.keys()):
                    if name in seen or index.get(name) != filename:
                        raise ValueError("Source index mismatch: " + name)
                    seen.add(name)
                    weight = handle.get_tensor(name)
                    if weight.dtype != torch.bfloat16:
                        raise ValueError("Expected BF16 source tensor: " + name)
                    if not weight.isfinite().all():
                        raise ValueError("Nonfinite source tensor: " + name)
                    layer = name.removesuffix(".weight")
                    if (
                        name.endswith(".weight")
                        and weight.ndim == 2
                        and selected_layer(layer, args.policy)
                    ):
                        codes, scales = quantize_weight(weight)
                        quant["quantized_layers"][layer] = {
                            "n": weight.shape[0],
                            "k": weight.shape[1],
                            "weight_codes": layer + ".weight_codes",
                            "weight_scale": layer + ".weight_scale",
                        }
                        add(layer + ".weight_codes", codes)
                        add(layer + ".weight_scale", scales)
                    else:
                        add(name, weight)
                flush()
            after = (source / filename).stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Source changed while exporting: " + filename)
        if seen != set(index):
            raise ValueError("Missing source tensors")
        expected = {"same112": 112, "middle196": 196, "all224": 224}[args.policy]
        if len(quant["quantized_layers"]) != expected:
            raise ValueError("Unexpected layer count")
        (destination / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                indent=2,
            )
            + "\n"
        )
        (destination / "quantization_config.json").write_text(
            json.dumps(quant, indent=2) + "\n"
        )
        component_hashes = {}
        for component in [
            "model_index.json",
            "configuration.json",
            "LICENSE",
            "text_encoder",
            "vae",
            "processor",
            "scheduler",
        ]:
            entry = args.source / component
            if not entry.exists():
                continue
            print("Copying unchanged component " + component, flush=True)
            files = sorted(entry.rglob("*")) if entry.is_dir() else [entry]
            for file in files:
                if not file.is_file() or any(
                    part.startswith(".") for part in file.relative_to(args.source).parts
                ):
                    continue
                relative = file.relative_to(args.source)
                target = args.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = sha256(file)
                shutil.copy2(file, target)
                if sha256(target) != digest:
                    raise ValueError("Component copy mismatch: " + str(relative))
                component_hashes[str(relative)] = digest
        report = {
            "source": str(args.source.resolve()),
            "policy": args.policy,
            "quantized_linears": expected,
            "payload_bytes": total_bytes,
            "source_sha256": source_hashes,
            "shard_sha256": hashes,
            "seconds": time.perf_counter() - start,
            "unquantized_component_sha256": component_hashes,
            "component_storage": "independent copies; original dtypes preserved",
            "serialized_tensors_verified": True,
            "precision_validation": "pending; export does not establish activation or image quality",
        }
        (args.output / "quantization_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        (args.output / "EXPORT_COMPLETE").write_text(
            "Portable per-channel W8 codes and FP32 scales; runtime A8; no opaque buffers\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        return report
    except BaseException as error:
        (args.output / "EXPORT_FAILED").write_text(str(error) + "\n")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=18)
    args = p.parse_args()
    if args.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(args.threads)
    export_checkpoint(args.source, args.output)


if __name__ == "__main__":
    main()
