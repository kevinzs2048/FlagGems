"""Measure plain PyTorch FP32 Linear scheduling on actual Qwen 2.1 weights."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

from .torch_baseline import FP32Linear, configure_linear_workers


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_interop_threads(1)
    torch.manual_seed(51)
    root = args.model / "transformer"
    index = json.loads(
        (root / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    rows = []
    for name in (
        "transformer_blocks.2.attn.to_q",
        "transformer_blocks.2.img_mlp.proj",
        "transformer_blocks.2.img_mlp.out",
    ):
        key = name + ".weight"
        with safe_open(root / index[key], framework="pt") as source:
            weight = source.get_tensor(key)
        original = torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=False, device="meta"
        )
        original.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer = FP32Linear(original)
        for m in (1024, 4096):
            x = torch.randn(m, weight.shape[1], dtype=torch.bfloat16)
            configure_linear_workers()
            expected = layer(x)
            for threads in (4, 8, 18):
                torch.set_num_threads(threads)
                for workers, axis in (
                    (1, 0),
                    (2, 0),
                    (3, 0),
                    (4, 0),
                    (6, 0),
                    (2, 1),
                    (3, 1),
                    (4, 1),
                    (6, 1),
                ):
                    configure_linear_workers(workers, axis)
                    layer(x)
                    times = []
                    for _ in range(3):
                        started = time.perf_counter()
                        actual = layer(x)
                        times.append(time.perf_counter() - started)
                    error = (actual.float() - expected.float()).abs()
                    row = dict(
                        layer=name,
                        m=m,
                        n=weight.shape[0],
                        k=weight.shape[1],
                        threads=threads,
                        workers=workers,
                        axis=axis,
                        seconds=times,
                        median_seconds=statistics.median(times),
                        max_abs=float(error.max()),
                        rel_l2=float(
                            torch.linalg.vector_norm(error)
                            / torch.linalg.vector_norm(expected.float())
                        ),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    configure_linear_workers()


if __name__ == "__main__":
    main()
