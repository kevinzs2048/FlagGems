"""PyTorch CPU pipeline loading; independent of Torch-FL and test harnesses."""

import json
import time
from pathlib import Path

import torch

from .w8a8_sme import DynamicW8SMELinear, load_transformer_w8


def tune(model, backend="native"):
    from qwen_image_cpu.arm_operator_backend import enable as enable_backend
    from qwen_image_cpu.attention_layout import enable_attention_layout
    from qwen_image_cpu.bf16_sme import enable_bf16_sme
    from qwen_image_cpu.pointwise_fusion import enable_swiglu
    from qwen_image_cpu.rmsnorm_fusion import enable_rmsnorm
    from qwen_image_cpu.rope_fusion import enable_rope
    from qwen_image_cpu.shared_modulation import enable_shared_modulation
    from qwen_image_cpu.sme_refinements import enable

    enable_bf16_sme(model, workers=18, axis=0, dynamic=True)
    enable_shared_modulation(model)
    for layer in model.modules():
        if isinstance(layer, DynamicW8SMELinear):
            layer.dynamic_tiles = True
            layer.direct_a8_pack = True
    assert enable_swiglu(model, packed=True) == 32
    enable_rope()
    enable_attention_layout()
    assert enable_rmsnorm(model) == 64
    enable(model, norm_rope=True, w8_swiglu=True, register_output=True, mlp_workers=12)
    provider = enable_backend(model, backend)
    for layer in provider.layers:
        layer.prepack()
    provider.prepack()
    return provider


@torch.inference_mode()
def load_pipeline(model_dir):
    from . import reference

    directory = Path(model_dir).expanduser().resolve()
    for name in (
        "model_index.json",
        "transformer/quantization_config.json",
        "text_encoder",
        "vae",
        "processor",
        "scheduler",
    ):
        if not (directory / name).exists():
            raise ValueError("Incomplete model directory: " + str(directory / name))
    quant = json.loads((directory / "transformer/quantization_config.json").read_text())
    if (
        quant.get("policy") != "same112"
        or len(quant.get("quantized_layers", {})) != 112
    ):
        raise ValueError("This inference profile requires the same112 W8A8 checkpoint")
    torch.set_num_threads(18)
    started = time.perf_counter()
    model, metadata = load_transformer_w8(
        directory, reference.QwenImage21Transformer2DModel
    )
    pipe = reference.QwenImage21Pipeline.from_pretrained(
        directory, transformer=model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cpu")
    reference.prepare_cpu_pipeline(pipe)
    provider = tune(model, "native")
    return (
        pipe,
        provider,
        {
            "load_prepack_seconds": time.perf_counter() - started,
            "reference_commit": reference.REFERENCE_COMMIT,
            "quantization": metadata,
            "framework": "PyTorch CPU",
            "torch_version": torch.__version__,
        },
    )


@torch.inference_mode()
def generate(
    pipe, provider, *, prompt, width=1024, height=1024, steps=40, seed=42, progress=None
):
    if (
        not prompt.strip()
        or min(width, height) < 32
        or width % 32
        or height % 32
        or steps < 1
    ):
        raise ValueError(
            "Nonempty prompt, positive steps and dimensions divisible by32 required"
        )
    latents, _ = pipe.prepare_latents(
        None,
        1,
        pipe.transformer.config.in_channels,
        height,
        width,
        pipe.transformer.dtype,
        torch.device("cpu"),
        torch.Generator(device="cpu").manual_seed(seed),
        latents=None,
    )
    provider.reset()

    def callback(p, step, t, kwargs):
        if progress:
            progress(step + 1, steps)
        return kwargs

    started = time.perf_counter()
    result = pipe(
        prompt=prompt,
        true_cfg_scale=1.0,
        num_inference_steps=steps,
        height=height,
        width=width,
        latents=latents,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="pil",
        use_kv_cache=True,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=[],
    )
    seconds = time.perf_counter() - started
    dispatch = provider.snapshot()
    if (
        dispatch["distinct_w8_modules"] != 112
        or dispatch["calls"].get("native_w8_gemm") != 112 * steps
    ):
        raise RuntimeError("Incomplete W8 operator coverage: " + str(dispatch))
    image = result.images[0]
    if image.size != (width, height):
        raise RuntimeError("Unexpected generated image size")
    return image, {"resident_seconds": seconds, "dispatch": dispatch}
