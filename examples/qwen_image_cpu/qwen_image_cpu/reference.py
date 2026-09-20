"""User-supplied Qwen-Image 2.1 reference with isolated local CPU compatibility."""

import os
import sys
from pathlib import Path

import huggingface_hub
import huggingface_hub.errors
import torch

from .cpu_accumulation import enable_fp32_accumulation

REFERENCE_COMMIT = "9511e982f22ea8fcc64faf277d90b3a21f8a83f6"
REFERENCE_SOURCE = (
    Path(os.environ.get("QWEN_IMAGE_DIFFUSERS_SRC", "__missing_reference_source__"))
    .expanduser()
    .resolve()
)
if not REFERENCE_SOURCE.is_dir():
    raise RuntimeError(
        "Set QWEN_IMAGE_DIFFUSERS_SRC to the reference Diffusers src directory"
    )
if "diffusers" in sys.modules:
    raise RuntimeError(
        "Load the reference adapter before importing another Diffusers version"
    )


def _remote_unavailable(*args, **kwargs):
    raise RuntimeError(
        "This compatibility adapter supports existing local model directories only"
    )


for _name in ("resolve_revision", "get_cached_repo_tree"):
    if not hasattr(huggingface_hub, _name):
        setattr(huggingface_hub, _name, _remote_unavailable)
for _name in ("RevisionResolutionError", "CachedRepoTreeNotFoundError"):
    if not hasattr(huggingface_hub.errors, _name):
        setattr(huggingface_hub.errors, _name, type(_name, (Exception,), {}))

sys.path.insert(0, str(REFERENCE_SOURCE))
import diffusers  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKLQwenImage21,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.models.autoencoders import (  # noqa: E402
    autoencoder_kl_qwenimage21 as vae_source,
)
from diffusers.models.transformers import (  # noqa: E402
    transformer_qwenimage21 as transformer_source,
)

__version__ = diffusers.__version__
_original_dispatch = transformer_source.dispatch_attention_fn


def _cpu_dispatch(query, key, value, *args, **kwargs):
    if query.device.type == "cpu" and query.dtype == torch.bfloat16:
        return _original_dispatch(
            query.float(), key.float(), value.float(), *args, **kwargs
        ).to(query.dtype)
    return _original_dispatch(query, key, value, *args, **kwargs)


class _VaeFunctional:
    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)

    def scaled_dot_product_attention(self, query, key, value, *args, **kwargs):
        if query.device.type == "cpu" and query.dtype == torch.bfloat16:
            return torch.nn.functional.scaled_dot_product_attention(
                query.float(), key.float(), value.float(), *args, **kwargs
            ).to(query.dtype)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, *args, **kwargs
        )


def prepare_cpu_pipeline(pipe):
    transformer_source.dispatch_attention_fn = _cpu_dispatch
    vae_source.F = _VaeFunctional()
    counts = {
        "transformer": enable_fp32_accumulation(pipe.transformer, cache_weights=True),
        "vae": enable_fp32_accumulation(pipe.vae, cache_weights=True),
        "text_encoder": (
            enable_fp32_accumulation(pipe.text_encoder, cache_weights=False)
            if pipe.text_encoder is not None
            else 0
        ),
    }
    if pipe.text_encoder is not None:
        pipe.text_encoder.forward = pipe.text_encoder.model.forward
    pipe.fp32_cpu_accumulation = True
    pipe.floating_weight_cache_policy = {
        "transformer_and_vae": "FP32 copies of unchanged BF16 values",
        "text_encoder": "transient FP32 per layer",
        "attention": "FP32 SDPA with BF16 input and output",
    }
    return {
        "reference_commit": REFERENCE_COMMIT,
        "source": str(REFERENCE_SOURCE),
        "patched_modules": counts,
        "architecture": "unaltered reference classes",
        "remote_loading": "disabled; local paths only",
    }


__all__ = [
    "QwenImage21Pipeline",
    "QwenImage21Transformer2DModel",
    "AutoencoderKLQwenImage21",
    "prepare_cpu_pipeline",
]
