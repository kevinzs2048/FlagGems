"""Generate an image with the W8A8 PyTorch CPU profile."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument(
        "--reference-source",
        type=Path,
        help="Reference Diffusers src directory; or QWEN_IMAGE_DIFFUSERS_SRC",
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("result.png"))
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--runs", type=int, default=1)
    args = p.parse_args(argv)
    if (
        not args.prompt.strip()
        or args.steps < 1
        or args.runs < 1
        or args.warmup_steps < 0
    ):
        p.error(
            "prompt must be nonempty, steps/runs positive and warmup-steps nonnegative"
        )
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32:
        p.error("width/height must be positive multiples of32")
    if args.output.suffix.lower() != ".png":
        p.error("output must end with .png")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.reference_source:
        os.environ["QWEN_IMAGE_DIFFUSERS_SRC"] = str(args.reference_source.resolve())
    import torch

    from .pipeline import generate, load_pipeline

    torch.set_num_interop_threads(1)
    pipe, provider, metadata = load_pipeline(args.model)
    metadata["configuration"] = {
        "model": str(args.model.resolve()),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "true_cfg_scale": 1.0,
        "kv_cache": True,
        "weight": "112 Linear W8A8, remaining BF16",
        "backend": "native SME2",
        "threads": 18,
        "mlp_workers": 12,
    }
    metadata["requests"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        prompt=args.prompt, width=args.width, height=args.height, seed=args.seed
    )
    if args.warmup_steps:
        print("Warmup", flush=True)
        _, warm = generate(pipe, provider, steps=args.warmup_steps, **kwargs)
        metadata["warmup"] = warm
    for run in range(args.runs):
        image, record = generate(pipe, provider, steps=args.steps, **kwargs)
        path = (
            args.output
            if args.runs == 1
            else args.output.with_name(f"{args.output.stem}-{run+1:02d}.png")
        )
        image.save(path)
        record.update(
            image=str(path.resolve()),
            pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        metadata["requests"].append(record)
        args.output.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        print(f'Saved {path}: {record["resident_seconds"]:.3f}s', flush=True)


if __name__ == "__main__":
    main()
