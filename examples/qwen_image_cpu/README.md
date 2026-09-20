# Qwen-Image PyTorch CPU example

The model application has moved to **flagos-macos-runtime**:

- `src/qwen_image_cpu/`: checkpoint loading, reference Diffusers CPU adapters,
  model fusion wiring and image generation.
- `bin/qwen-image`: source-checkout launcher.
- `docs/qwen_image/MODEL_CARD_MAC.md`: English Mac model card.
- `tests/qwen_image/`: model integration and regression tests.

The runtime repository is https://github.com/kevinzs2048/flagos-macos-runtime.
Use its matching Qwen-Image development revision; these changes are not yet in
an existing public runtime release.

Install the runtime adapter into the prepared Python environment. This directory
only retains `run.sh`, a compatibility launcher calling that installed package:

```bash
bash run.sh --model /path/to/model --prompt 'A watercolor landscape' \
  --width 1024 --height 1024 --steps 40 --output result.png
```

Generic computation is owned by FlagGems:

- `src/flag_gems/runtime/backend/_arm/quantized_linear/sme2/`: Linear modules,
  packing, cache management and execution policies.
- `src/flag_gems/runtime/backend/_arm/ops/`: Triton operators.
- `src/flag_gems/csrc/arm/qwen_image/`: existing native C++/assembly kernels.
- `tests/arm/sme2/`: model-independent operator tests.

FlagGems has no dependency on the runtime model package or reference Diffusers.
FlagTree-CPU owns compiler/lowering support. Neither PyTorch source nor
Torch-FL integration is required for this application.
