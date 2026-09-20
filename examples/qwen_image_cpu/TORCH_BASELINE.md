# Plain PyTorch CPU performance baseline

This opt-in benchmark runs the user-supplied Qwen-Image-2.1 reference on CPU
without importing FlagGems, installing Torch-FL, compiling an extension, or
calling our KleidiAI/SME2/Triton operators. PyTorch's own Apple Accelerate BLAS
and CPU SDPA are allowed. Accelerate may use hardware acceleration internally;
this benchmark does not claim to disable the CPU's matrix hardware.

## Precision and memory

Use the **original BF16 checkpoint**, not the W8A8 export. Transformer Linear
weights are exactly promoted to FP32 once. FP32 computation returns BF16
activations at every Linear boundary. Norms and the reference model structure
are preserved. CPU SDPA and VAE convolutions use the existing pure-ATen FP32
compatibility adapter. Text-encoder Linear weights are promoted transiently.

This is **BF16-valued weights / FP32 computation / BF16 boundaries**. It is
neither native BF16 GEMM nor W8A8. Keep that distinction when comparing with
the 112-layer W8A8 production profile.

`--storage promote` keeps one FP32 copy of Transformer Linear weights, avoiding
the memory overhead of retaining both BF16 weights and FP32 caches. `cache`
and `transient` are available for separate memory/performance experiments.

Optional `--linear-workers` uses a persistent Python ThreadPoolExecutor to
submit independent ordinary `torch.nn.functional.linear` calls. Axis 0 splits
input rows; axis 1 splits output channels. There is no custom native code.
Small projections keep their normal single-call implementation. The 162-case
real-weight microbenchmark on this machine found identical BF16 outputs for
all tested scheduling configurations; this is not a universal numerical
guarantee for every matrix shape or PyTorch version.

## Reproduction

Use the same PyTorch environment as the optimized pipeline, but expose only
this example directory on PYTHONPATH. Set QWEN_IMAGE_DIFFUSERS_SRC to the frozen
reference Diffusers `src` directory. Do not source a FlagOS backend setup file.

```bash
export PYTHONPATH="$PWD/examples/qwen_image_cpu"
export QWEN_IMAGE_DIFFUSERS_SRC=/path/to/reference/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -m qwen_image_cpu.torch_gemm_probe \
  --model "$HOME/Qwen-Image-2.1" --output gemm-probe.json

python -m qwen_image_cpu.torch_baseline \
  --model "$HOME/Qwen-Image-2.1" --prompt 'Your prompt' \
  --linear-workers 6 --linear-axis 1 --threads 8 18 \
  --scan-size 1024 --scan-repeats 1 \
  --sizes 512 1024 --steps 40 --runs 2 --output torch-baseline
```

Thread selection uses the median cached-prefix Transformer forward time from
three-step real model requests, excluding each request's first forward.
Selection is a measured search over the listed candidates, not proof of a
global optimum. The selected configuration then runs every complete request.

## Timing boundaries

- Batch 1, CFG 1, reference prefix KV cache enabled, seed 42.
- Report resident request latency: prompt encoding, all denoising steps,
  scheduler, VAE and conversion to the returned PIL image.
- Exclude model loading, initial latent construction, PNG encoding and file IO.
- Prompt encoding occurs on every request; no prompt cache substitutes for it.
- Record each Transformer call, stage totals, image hash, backend audit and
  configuration in `report.json`; save actual PNGs for every full request.
- A short scan is not an end-to-end measurement or a 40-step quality test.

The separate optimized production baseline remains unchanged. Completed
40-step measurements and their scope are in `INTEGRATION_REPORT.md`.
The plain PyTorch measurements and image checks are in
[TORCH_BASELINE_RESULTS.md](TORCH_BASELINE_RESULTS.md).
