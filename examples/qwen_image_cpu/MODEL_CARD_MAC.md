# Qwen-Image-2.1-0920 W8A8 — PyTorch CPU on Mac

## Introduction

This checkpoint runs Qwen-Image-2.1 text-to-image generation on Apple M5 Pro
using **PyTorch CPU** and FlagGems-owned SME2/KleidiAI operators. The text
encoder, diffusion Transformer and VAE all run on the CPU.

The Transformer uses mixed precision: 112 Linear modules use W8A8, and the
remaining Transformer weights retain BF16. Text Encoder weights retain BF16.
VAE files retain the original FP32 storage and are loaded as BF16 for inference;
selected CPU operations, including VAE convolution, compute in FP32.

| Component | Weight file size, approximately |
| --- | ---: |
| Mixed W8A8/BF16 Transformer | 10.48 GB |
| BF16 Text Encoder | 17.53 GB |
| FP32 VAE | 1.35 GB |
| Complete model directory | 29.38 GB |

Sizes use decimal GB. INT8 weights use symmetric per-output-channel quantization
with FP32 scales. Activations use dynamic asymmetric per-token INT8 quantization.

## Mac Requirements

Validated hardware: **Apple M5 Pro, 18 CPU cores, 64 GiB memory, macOS arm64**.
The native operator profile requires SME2, BF16 matrix instructions and the
validated 512-bit streaming vector layout. Other Mac processors have not been
validated with this profile.

Validated software:

- Python 3.11 and unmodified PyTorch 2.11.0.
- FlagGems with the Qwen-Image CPU adapter and native Arm operators.
- The supplied Qwen-Image-2.1 Diffusers reference implementation, revision
  `9511e982f22ea8fcc64faf277d90b3a21f8a83f6`.
- Transformers 4.57.6, Hugging Face Hub 0.36.2, Safetensors, Accelerate and Pillow.
- KleidiAI 1.29.0, including `build-perf/libkleidiai.a`.
- The matching FlagTree-CPU build and compatible OpenMP runtime from the
  validated environment; Apple command-line developer tools for extension builds.

This release currently uses a prepared source environment. A public installer
or downloadable runtime package for this Qwen profile has not been published.
The reference Diffusers source and local FlagGems changes must accompany the
environment; installing stock PyTorch and a generic Diffusers release alone
does not provide this checkpoint loader. The default measured kernels are
native SME2/KleidiAI kernels.

## Run Inference

### 1. Prepare the environment and model directory

Use the complete checkpoint directory, including `transformer`, `text_encoder`,
`vae`, `processor`, `scheduler` and `model_index.json`.

On the validated development Mac, the existing environment and model are:

```bash
source "$HOME/qwen-image-2.1-repos/env-upstream.sh"
export FLAGGEMS_DIR="$HOME/qwen-image-2.1-repos/FlagGems-master"
export MODEL_DIR="$HOME/Qwen-Image-2.1-0920-W8A8-PerChannel-112"
```

For another prepared Mac environment, set `FLAGGEMS_DIR` and `MODEL_DIR` to
the corresponding locations. Its environment setup must also supply
`QWEN_IMAGE_PYTHON`, `QWEN_IMAGE_DIFFUSERS_SRC`, `FLAGGEMS_KLEIDIAI_ROOT` and
the matching compiler/OpenMP configuration. `QWEN_IMAGE_DIFFUSERS_SRC` points
to the reference implementation's `src` directory.

### 2. Generate an image

```bash
bash "$FLAGGEMS_DIR/examples/qwen_image_cpu/run.sh" \
  --model "$MODEL_DIR" \
  --prompt 'A capybara reading a book by candlelight, watercolor painting' \
  --width 1024 --height 1024 \
  --steps 40 --seed 42 \
  --output result.png
```

The launcher loads the checkpoint, prepares the CPU operators, runs text
encoding, denoising and VAE decoding, and saves the image. First startup may
include native extension compilation. No PyTorch source patch is required.

Outputs:

- `result.png`: generated image.
- `result.json`: generation settings, load/prepack time, resident request time,
  image hashes and executed W8 operator counts.

The validated profile uses CFG=1, prefix KV caching, 18 main threads and 12 MLP
workers. Width and height must be positive multiples of 32. For a smaller
functional check, use `--width 512 --height 512`.

### 3. Measure resident inference time

Add `--warmup-steps 2` to run a two-step warmup before the measured request.
To repeat requests in one process, also add `--runs 3`; outputs are named
`result-01.png`, `result-02.png` and `result-03.png`.

## Measured Performance

New September 20 weights, one prompt, seed 42, CFG=1, a two-step warmup:

| Resolution | Denoising steps | Resident request time |
| --- | ---: | ---: |
| 512 × 512 | 40 | 158.76 seconds |
| 1024 × 1024 | 40 | 811.28 seconds |

Times include text encoding, denoising, VAE decoding and PIL conversion. They
exclude model loading, prepacking, warmup, initial latent construction and PNG
writing. Each row is a single measured request. Both requests executed 112
distinct W8 Linear modules and 4,480 W8 GEMMs. These measurements are image
generation latency, rather than language-model token throughput.

## Validation

The export and numerical validation regression suite passed 36 tests. Twelve
sampled real-input kernel checks passed against an integer reference. Numerical
differences from BF16 were measured on identical Transformer inputs at steps
1, 20 and 40; quantization is not lossless.

Image checks currently cover one prompt. Images were generated successfully at
both resolutions; the 1024 image has recognizable text with imperfect letter
shapes. A full 100-prompt evaluation of these new weights is not complete.

When reading this card in the model directory:

- [1024 example image](validation/1024-40.png)
- [512 example image](validation/512-40.png)
- [Numerical results](validation/precision.json)
- [1024 timing and operator coverage](validation/1024-40.json)
- [512 timing and operator coverage](validation/512-40.json)

## Inference Architecture

```text
run.sh
  -> Python application: load model, generate image, save results
  -> Qwen-Image-2.1 Diffusers pipeline
  -> PyTorch CPU with FlagGems Arm operators
  -> SME2/KleidiAI and ATen CPU computation
```

The Python application supplies model-specific loading and CPU adapters for
this W8 checkpoint. PyTorch itself is unmodified. The workflow requires no
vLLM, vLLM-Plugin-FL or Torch-FL service.
