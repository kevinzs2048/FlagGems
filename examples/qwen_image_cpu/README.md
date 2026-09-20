# Qwen-Image-2.1 W8A8 — PyTorch CPU inference on Mac

This example runs Qwen-Image-2.1 on **PyTorch CPU**, with FlagGems-owned native
SME2/KleidiAI operators. The validated profile quantizes 112 Transformer Linear
modules to W8A8; other weights remain BF16. It requires no Torch-FL runtime or
`flagos` device registration.

## Files and ownership

- `qwen_image_cpu/`: model loading, checkpoint policy, fusion adapters and CLI.
- `../../src/flag_gems/csrc/arm/qwen_image/`: native C++/assembly operators.
- `../../src/flag_gems/runtime/backend/_arm/quantized_linear/image_sme.py`:
  native operator compilation/registration, with cached loads.
- FlagGems `_arm/ops/w8a8_*sme2.py`: optional Triton operator candidates.
- Triton CPU lowering changes are maintained in the separate compiler repository.

The default execution path preserves the original native kernels, integer
quantization, floating-point rounding and tuned 18-thread/12-MLP-worker profile.

## Environment

Validated target: Apple M5 Pro, macOS arm64, SME2 with 512-bit streaming vectors,
Python 3.11, PyTorch 2.11.0, Transformers 4.57.6, KleidiAI 1.29.0, AppleClang and
OpenMP. Text Encoder and VAE run on CPU as well. BF16 describes weight/storage
and boundary dtypes; selected floating operators compute/accumulate in FP32.

Use an environment containing PyTorch CPU and the locked FlagGems/Triton CPU
build. The matching Qwen-Image-2.1 implementation is required: the reference
Diffusers archive identifies commit 9511e982f22ea8fcc64faf277d90b3a21f8a83f6.
An arbitrary PyPI Diffusers release cannot replace that model implementation.

```bash
export QWEN_IMAGE_PYTHON=/path/to/python3.11
export QWEN_IMAGE_DIFFUSERS_SRC=/path/to/Qwen-Image-2.1-reference/src
export FLAGGEMS_KLEIDIAI_ROOT=/path/to/KleidiAI-v1.29.0
```

KleidiAI must contain `build-perf/libkleidiai.a`. Native extensions compile on
first use; this requires the developer tools and compatible OpenMP headers.
The example does not install packages into the host Python or choose another
quantization scheme automatically.

## Generate an image

From this directory:

```bash
./run.sh \
  --model /path/to/Qwen-Image-2.1-W8A8-PerChannel-112 \
  --prompt 'A capybara wearing a wizard hat, reading a book by candlelight, oil painting' \
  --width 1024 --height 1024 --steps 40 --seed 42 \
  --output result.png
```

`--model` is one complete pipeline directory: W8 Transformer plus BF16 Text
Encoder/VAE, processor, scheduler and model_index.json. The checkpoint contract
is symmetric per-output-channel INT8 weights with FP32 scales and dynamic
asymmetric per-token INT8 activations. The loader requires the same112 policy.
For release packaging, include actual component files; developer symlinks back
to another local checkpoint are not a downloadable model artifact.

Output: `result.png` and `result.json`. The JSON records configuration, source
revision, load/prepack time, resident request time and executed W8 operator
coverage. Every generated image requires 112 distinct W8 modules and 112×steps
W8 GEMMs. CFG is 1 and prefix KV caching is enabled.

For repeated resident timing, add `--warmup-steps 40 --runs 3`. Loading and warmup
are reported separately; image writing is excluded from resident request time.
Do not compare this with an LLM PP512/TG128 benchmark.

## Tests

With the same environment and the example directory on PYTHONPATH:

```bash
python -m pytest tests -q --tb=short
```

Tests exercise integer oracles, packing, fusion correctness, native/Triton
comparison and input validation. Full image regression and measured timing are
recorded separately in the integration report.

## Repository baselines and dependencies

- FlagGems base: official `master`, `df6fcaad6c818312ad3593e6213120723e1a2f26`.
- FlagTree-CPU base: official `triton_v3.7.x`,
  `2c35990a30e96665f8f9b5e158562288b4011048`; SME2 update `de687051c`.
- KleidiAI: unmodified v1.29.0, `13cd359`; source and static library required.
- Diffusers model implementation: unmodified user-provided reference archive,
  revision `9511e982f22ea8fcc64faf277d90b3a21f8a83f6`. Its original fork URL
  has not been established. Supply that exact source separately.
- PyTorch: unmodified 2.11.0. No PyTorch patch, pytorch-plugin-FL or Torch-FL
  registration is required.

Install the source-built FlagTree-CPU in the Python environment, then install
this FlagGems checkout with `python -m pip install --no-deps -e .` from its root.
The `--no-deps` option preserves the selected CPU compiler. The example also
needs Accelerate, Safetensors, Transformers, Hugging Face Hub and Pillow, plus
the dependencies of the reference Diffusers source. The tested Hub version is
0.36.2; this adapter intentionally supports local checkpoints only.

For macOS OpenMP, set `TRITON_LOCAL_LIBOMP_PATH` and `TRITON_SYS_PATH` to a prefix
with `include/omp.h` and `lib/libomp.dylib`, using the same OpenMP runtime as
PyTorch. This configuration belongs in the environment/launcher. GNU-specific
settings such as `GOMP_SPINCOUNT` are not a substitute for LLVM OpenMP tuning.

The native operator implementation lives in FlagGems C++/assembly sources.
Python model-specific fusion wiring lives in this repository's example package.
Only `run.sh` is a shell launcher. Optional exploration modules are retained
for numerical regression; their presence does not enable them in production.
Model weights, native build caches and the reference archive are not committed.

## Validation reports

- [Integration, image parity and end-to-end timing](INTEGRATION_REPORT.md)
- [stable-diffusion.cpp compatibility probe](STABLE_DIFFUSION_CPP_REVIEW.md)
