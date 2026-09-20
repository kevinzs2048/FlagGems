"""Separate BF16 backend, W8 weight, A8 and packing errors on a fresh trajectory."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch

from .torch_baseline import promote_linears, request
from .w8a8_sme import (
    DynamicW8SMELinear,
    activation_reference,
    load_transformer_w8,
    ops,
    unpack_activation,
)


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    delta = a - b
    if not a.isfinite().all() or not b.isfinite().all():
        raise ValueError("Nonfinite prediction")
    return dict(
        relative_l2=float(delta.norm() / b.norm().clamp_min(1e-12)),
        max_abs=float(delta.abs().max()),
        cosine=float(
            torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        ),
        bitwise_equal=bool(torch.equal(actual, expected)),
    )


@torch.inference_mode()
def check_integer_oracle(layer, x):
    """Full K, two captured rows, 64 spread output channels; integer PyTorch oracle."""
    native = ops()
    m, k = x.shape
    lhs = native.quant_pack(x)
    codes, scale, zero = unpack_activation(lhs, m, k)
    expected_codes, expected_scale, expected_zero = activation_reference(x)
    assert torch.equal(codes, expected_codes) and torch.equal(zero, expected_zero)
    torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=0)
    indices = torch.linspace(
        0, layer.out_features - 1, min(64, layer.out_features)
    ).long()
    integer = (codes.int() - zero[:, None]) @ layer.weight_codes[indices].int().T
    expected = integer.float() * scale[:, None] * layer.weight_scale[indices][None, :]
    actual = native.matmul(lhs, layer.prepack(), m, layer.out_features, k, 18, 0)[
        :, indices
    ]
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-5)
    return metrics(actual, expected)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--quantized", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(18)
    torch.set_num_interop_threads(1)
    from . import reference
    from .bf16_sme import enable_bf16_sme

    report = {
        "source": str(args.source),
        "quantized": str(args.quantized),
        "scope": (
            "one prompt; fresh BF16 SME2 512/40 trajectory; identical captured inputs, "
            "prefix recomputed in every comparison"
        ),
        "stages": {},
        "integer_oracle": [],
        "status": "running",
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    pipe = reference.QwenImage21Pipeline.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cpu")
    reference.prepare_cpu_pipeline(pipe)
    enable_bf16_sme(pipe.transformer, workers=18, axis=0, dynamic=True)
    captures = {}
    step = 0

    def capture(module, positional, kwargs):
        nonlocal step
        step += 1
        if step in (1, 20, 40):
            captures[step] = copy.deepcopy(
                {
                    key: value
                    for key, value in kwargs.items()
                    if key not in ("kv_cache", "kv_cache_mode")
                }
            )
            torch.save(captures[step], args.output / f"input-{step:02d}.pt")

    handle = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    image, timing = request(pipe, args.prompt, 512, 40, 42)
    handle.remove()
    image.save(args.output / "bf16-sme-512-40.png")
    timing.pop("linear_workers", None)
    timing.pop("linear_axis", None)
    timing["backend"] = "BF16 SME2"
    report["bf16_trajectory"] = timing
    assert set(captures) == {1, 20, 40}
    model = pipe.transformer
    del pipe
    gc.collect()
    predictions = {}

    def evaluate(stage):
        predictions[stage] = {}
        report["stages"][stage] = []
        for step, kwargs in captures.items():
            started = time.perf_counter()
            value = model(**kwargs)[0][:, -1024:].clone()
            elapsed = time.perf_counter() - started
            if not value.isfinite().all():
                raise ValueError("Nonfinite " + stage)
            predictions[stage][step] = value
            row = dict(
                step=step, timestep=float(kwargs["timestep"].item()), seconds=elapsed
            )
            for other in ("bf16_fp32", "w8a16"):
                if other in predictions and other != stage:
                    row["vs_" + other] = metrics(value, predictions[other][step])
            report["stages"][stage].append(row)
            torch.save(value, args.output / f"{stage}-{step:02d}.pt")
            print(json.dumps(dict(stage=stage, **row)), flush=True)
            save()

    evaluate("bf16_sme")
    # The replacement retains the exact BF16 values and uses ATen FP32 GEMM.
    # Collect replaced modules so their packed caches cannot cause memory pressure.
    for child in model.modules():
        if hasattr(child, "_bf16_sme_packed"):
            child._bf16_sme_packed = None
    promote_linears(model)
    gc.collect()
    evaluate("bf16_fp32")
    for row in report["stages"]["bf16_sme"]:
        step = row["step"]
        row["vs_bf16_fp32"] = metrics(
            predictions["bf16_sme"][step], predictions["bf16_fp32"][step]
        )
    del model
    gc.collect()
    model, metadata = load_transformer_w8(
        args.quantized, reference.QwenImage21Transformer2DModel
    )
    report["checkpoint"] = metadata
    promote_linears(model)
    layers = [
        layer for layer in model.modules() if isinstance(layer, DynamicW8SMELinear)
    ]
    assert len(layers) == 112
    for layer in layers:
        layer.activation_quantization = False
    evaluate("w8a16")
    for layer in layers:
        layer.activation_quantization = True
        layer.dynamic_tiles = True
        layer.direct_a8_pack = True
    names = (
        "transformer_blocks.2.attn.to_q",
        "transformer_blocks.15.img_mlp.proj",
        "transformer_blocks.29.img_mlp.gate_layer",
        "transformer_blocks.29.attn.to_k",
    )
    samples = []
    handles = []
    for name in names:

        def capture_activation(layer, inputs, name=name):
            value = inputs[0].reshape(-1, layer.in_features)
            samples.append((name, value[[0, value.shape[0] - 1]].clone()))

        handles.append(
            model.get_submodule(name).register_forward_pre_hook(capture_activation)
        )
    evaluate("w8a8")
    for handle in handles:
        handle.remove()
    for name, x in samples:
        result = check_integer_oracle(model.get_submodule(name), x)
        report["integer_oracle"].append(dict(layer=name, **result))
    assert len(report["integer_oracle"]) == 12
    report["status"] = "complete"
    report["quality_scope"] = (
        "Numerical checks on one prompt, not dataset-wide image-quality acceptance"
    )
    save()


if __name__ == "__main__":
    main()
