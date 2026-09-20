# Re-export Qwen-Image-2.1 as same112 W8A8

The supported production policy selects four Linear weights in each block
with zero-based index 2 through 29: `attn.to_q`, `attn.to_k`, `img_mlp.proj`,
and `img_mlp.gate_layer`. This is 112 Linear modules in a 32-block model.

## Numerical format

- Original BF16 Transformer weights, quantized once.
- Symmetric signed INT8 weights, codes -127 through 127, zero point 0.
- One FP32 scale per output channel: `max(abs(row)) / 127`.
- Weight rounding: nearest even. An all-zero row has scale 1.
- Runtime dynamic asymmetric INT8 activations, one FP32 scale per token/row.
- Activation zero point rounding: nearest even. Scaled activation code rounding:
  nearest with ties away from zero, matching the KleidiAI 1.29.0 reference.
- INT32 accumulation, FP32 scaling/bias and BF16 output.
- Other Transformer weights remain BF16. Text encoder and VAE files are copied
  unchanged, retaining their source storage dtype. The September 20 source VAE
  is FP32 on disk; the existing pipeline loads it as BF16 and uses FP32 CPU
  convolution accumulation. No VAE INT8 quantization is performed.

This is per-channel W8A8 with FP32 scales, not G32 with BF16 scales.
Only portable codes/scales are serialized. CPU-specific packing happens at load
time. No static activation calibration, GPTQ, AWQ or rotation is used.

## Export

From the FlagGems repository with torch and safetensors installed:

```bash
PYTHONPATH=examples/qwen_image_cpu python -m qwen_image_cpu.export_w8 \
  --source "$HOME/Qwen-Image-2.1-0920" \
  --output "$HOME/Qwen-Image-2.1-0920-W8A8-PerChannel-112"
```

The exporter refuses existing output directories and already-quantized sources.
It verifies the source index, dtype, finite values and selected layer count;
reads every serialized tensor back for exact comparison; and verifies SHA-256
hashes for independent copies of unquantized components. Incomplete exports
are rejected by the runtime loader. `EXPORT_COMPLETE` means serialization
finished; it does not itself certify image quality.

## Validation

With the CPU inference environment configured as in README.md:

```bash
python -m qwen_image_cpu.validate_w8 \
  --source "$HOME/Qwen-Image-2.1-0920" \
  --quantized "$HOME/Qwen-Image-2.1-0920-W8A8-PerChannel-112" \
  --prompt 'Your prompt' --output validation-new-weights
```

The validator generates a fresh 512×512, 40-step BF16 SME2 trajectory. It
captures inputs at steps 1, 20 and 40, then recomputes the prefix on every
comparison branch using identical latents, embeddings and timesteps:

1. BF16-valued ATen FP32 Linear versus BF16 SME2: backend arithmetic differences.
2. W8A16 versus BF16-valued ATen FP32: weight quantization differences.
3. W8A8 versus W8A16: incremental activation quantization differences.
4. Captured A8 inputs and real W8 weights versus an exact INT32 PyTorch oracle:
   packing, activation quantization and kernel implementation correctness.

The oracle checks four representative Q/MLP layers across the three captured
steps, two actual input rows and 64 spread output channels per check, with the
full reduction dimension. It does not claim exhaustive coverage of every
element. The ordinary 512/40 and 1024/40 inference entry should also be run on
the exported model. Numerical errors and one prompt do not replace a broader
image-quality evaluation set.

## VAE storage and future quantization

The September 20 VAE contains 1,350,961,616 bytes of FP32 tensor data. Saving
those tensors as BF16 would use 675,480,808 bytes, excluding file headers.
This would reduce download/storage size, but is not expected to accelerate the
current pipeline: it already loads BF16 weights and promotes convolution
inputs/weights to FP32 for CPU computation. FP32 weight caches can also remain
resident. BF16 storage alone does not eliminate those caches.

VAE INT8 is a separate optimization requiring suitable convolution kernels,
activation range evaluation and comparisons decoding the same fixed latents.
Its end-to-end benefit must be measured against the actual VAE fraction of
request time. The VAE runs after the denoising loop. This export preserves the
original VAE files and makes no VAE INT8 accuracy or speed claim.
