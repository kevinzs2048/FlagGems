"""CPU-only ATen baseline; no FlagGems, Triton, or custom native operators.

BF16 checkpoint values are exactly promoted to FP32 for Linear computation.
Linear outputs and the model's activation boundaries remain BF16. This is
explicitly a BF16-valued/FP32-compute baseline, not a W8A8 implementation.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.nn import functional as F


class FP32Linear(torch.nn.Module):
    """Keep a single FP32 copy instead of BF16 weights plus an FP32 cache."""

    executor = None
    workers = 1
    axis = 0

    def __init__(self, layer):
        super().__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.weight = torch.nn.Parameter(layer.weight.float(), requires_grad=False)
        self.bias = (
            torch.nn.Parameter(layer.bias.float(), requires_grad=False)
            if layer.bias is not None
            else None
        )

    def forward(self, x):
        value = x.float()
        if (
            self.executor is None
            or value.numel() // self.in_features < 256
            or self.out_features < 1024
        ):
            return F.linear(value, self.weight, self.bias).to(x.dtype)
        flat = value.reshape(-1, self.in_features)
        if self.axis == 0:
            futures = [
                self.executor.submit(F.linear, part, self.weight, self.bias)
                for part in flat.chunk(self.workers, dim=0)
            ]
            result = torch.cat([future.result() for future in futures], dim=0)
        else:
            weights = self.weight.chunk(self.workers, dim=0)
            biases = (
                self.bias.chunk(self.workers)
                if self.bias is not None
                else [None] * len(weights)
            )
            futures = [
                self.executor.submit(F.linear, flat, weight, bias)
                for weight, bias in zip(weights, biases)
            ]
            result = torch.cat([future.result() for future in futures], dim=1)
        return result.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def configure_linear_workers(workers=1, axis=0):
    if FP32Linear.executor is not None:
        FP32Linear.executor.shutdown(wait=True)
    FP32Linear.workers = workers
    FP32Linear.axis = axis
    FP32Linear.executor = (
        ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    )


def promote_linears(model):
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, FP32Linear(child))
            count += 1
        else:
            count += promote_linears(child)
    return count


def audit(pipe):
    forbidden = [
        name
        for name in sys.modules
        if name == "flag_gems" or name.startswith("flag_gems.")
    ]
    libraries = sorted(str(path) for path in torch.ops.loaded_libraries)
    custom = [
        path
        for path in libraries
        if any(
            word in Path(path).name.lower()
            for word in ("qwen", "kleidi", "image_sme", "flaggems")
        )
    ]
    devices = sorted({str(p.device) for p in pipe.transformer.parameters()})
    if forbidden or custom or devices != ["cpu"]:
        raise RuntimeError(f"Non-baseline execution: {forbidden}, {custom}, {devices}")
    return {
        "flag_gems_imports": forbidden,
        "loaded_operator_libraries": libraries,
        "transformer_devices": devices,
        "torch_version": torch.__version__,
        "torch_build": torch.__config__.show(),
    }


@torch.inference_mode()
def load(model, storage):
    from . import reference

    started = time.perf_counter()
    pipe = reference.QwenImage21Pipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cpu")
    reference.prepare_cpu_pipeline(pipe)
    promoted = promote_linears(pipe.transformer) if storage == "promote" else 0
    if storage == "transient":
        for layer in pipe.transformer.modules():
            if getattr(layer, "_fp32_accumulation_enabled", False):
                layer._fp32_cache_weights = False
    pipe.set_progress_bar_config(disable=True)
    return pipe, {
        "load_seconds": time.perf_counter() - started,
        "promoted_linears": promoted,
        "reference_commit": reference.REFERENCE_COMMIT,
        "precision": "original BF16 values; FP32 Linear/Conv/SDPA; BF16 boundaries",
        "storage": storage,
        "audit": audit(pipe),
    }


@torch.inference_mode()
def request(pipe, prompt, size, steps, seed):
    # Match the optimized benchmark: explicit initial latents outside the timer.
    latents, _ = pipe.prepare_latents(
        None,
        1,
        pipe.transformer.config.in_channels,
        size,
        size,
        torch.bfloat16,
        torch.device("cpu"),
        torch.Generator(device="cpu").manual_seed(seed),
        latents=None,
    )
    stages = {"transformer": [], "text_encoder": [], "vae": []}
    handles = []
    for name, module in (
        ("transformer", pipe.transformer),
        ("text_encoder", pipe.text_encoder),
        ("vae", pipe.vae.decoder),
    ):
        starts = []

        def before(mod, args, stack=starts):
            stack.append(time.perf_counter())

        def after(mod, args, out, stack=starts, records=stages[name]):
            records.append(time.perf_counter() - stack.pop())

        handles.extend(
            (
                module.register_forward_pre_hook(before),
                module.register_forward_hook(after),
            )
        )

    def progress(p, step, timestep, values):
        print(
            json.dumps(
                {
                    "event": "step",
                    "size": size,
                    "step": step + 1,
                    "steps": steps,
                    "forward_seconds": stages["transformer"][-1],
                }
            ),
            flush=True,
        )
        return values

    started = time.perf_counter()
    try:
        result = pipe(
            prompt=prompt,
            true_cfg_scale=1.0,
            num_inference_steps=steps,
            height=size,
            width=size,
            latents=latents,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="pil",
            use_kv_cache=True,
            callback_on_step_end=progress,
            callback_on_step_end_tensor_inputs=[],
        )
        elapsed = time.perf_counter() - started
    finally:
        for handle in handles:
            handle.remove()
    if len(stages["transformer"]) != steps:
        raise RuntimeError("Unexpected number of Transformer forwards")
    image = result.images[0]
    if image.size != (size, size):
        raise RuntimeError("Unexpected image dimensions")
    return image, {
        "resident_seconds": elapsed,
        "stage_seconds": {name: sum(values) for name, values in stages.items()},
        "forward_seconds": stages["transformer"],
        "threads": torch.get_num_threads(),
        "linear_workers": FP32Linear.workers,
        "linear_axis": FP32Linear.axis,
        "size": size,
        "steps": steps,
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage", choices=("promote", "cache", "transient"), default="promote"
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 8, 12, 18])
    parser.add_argument("--scan-size", type=int, default=512)
    parser.add_argument("--linear-workers", type=int, default=1)
    parser.add_argument("--linear-axis", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scan-repeats", type=int, default=2)
    parser.add_argument("--sizes", nargs="*", type=int, default=[512, 1024])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        min(
            args.threads
            + [args.steps, args.runs, args.scan_repeats, args.linear_workers]
        )
        < 1
    ):
        parser.error("threads, steps, runs and scan-repeats must be positive")
    if any(s < 32 or s % 32 for s in args.sizes + [args.scan_size]):
        parser.error("image sizes must be positive multiples of 32")
    if not args.prompt.strip():
        parser.error("prompt must be nonempty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_interop_threads(1)
    torch.set_num_threads(18)
    if args.linear_workers > 1 and args.storage != "promote":
        parser.error("parallel Linear requires --storage promote")
    configure_linear_workers(args.linear_workers, args.linear_axis)
    pipe, report = load(args.model, args.storage)
    report.update(
        configuration={**vars(args), "output": str(args.output)},
        scan=[],
        requests=[],
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("OMP_", "VECLIB_", "OPENBLAS_"))
        },
    )

    def save():
        report["audit_final"] = audit(pipe)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    # Alternating sweep order reduces a simple warmup/order bias.
    for sweep in range(args.scan_repeats):
        for threads in args.threads if sweep % 2 == 0 else reversed(args.threads):
            torch.set_num_threads(threads)
            _, record = request(pipe, args.prompt, args.scan_size, 3, args.seed)
            report["scan"].append(record)
            save()
    scores = {
        threads: statistics.median(
            t
            for row in report["scan"]
            if row["threads"] == threads
            for t in row["forward_seconds"][1:]
        )
        for threads in args.threads
    }
    best = min(scores, key=scores.get)
    report["thread_scores"] = scores
    report["selected_threads"] = best
    torch.set_num_threads(best)
    save()
    for size in args.sizes:
        for run in range(args.runs):
            image, record = request(pipe, args.prompt, size, args.steps, args.seed)
            path = args.output / f"torch-{size}-{args.steps}-{run + 1:02d}.png"
            image.save(path)
            record["image"] = str(path)
            report["requests"].append(record)
            save()
    print(
        json.dumps(
            {"report": str(args.output / "report.json"), "selected_threads": best}
        ),
        flush=True,
    )
    configure_linear_workers()


if __name__ == "__main__":
    main()
