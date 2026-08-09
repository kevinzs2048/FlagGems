from .fused_add_rms_norm import fused_add_rms_norm  # noqa: F401


# Keep the legacy Qwen MLP patch lazy.  Importing it eagerly also imports its
# optional ``tle_ops`` fallback, which makes unrelated ordinary-Triton fused
# operators unusable with a stock Triton-CPU frontend.
def patch_qwen3_mlp(*args, **kwargs):
    from .patch_qwen3_mlp import patch_qwen3_mlp as implementation

    return implementation(*args, **kwargs)


def unpatch_qwen3_mlp(*args, **kwargs):
    from .patch_qwen3_mlp import unpatch_qwen3_mlp as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "fused_add_rms_norm",
    "patch_qwen3_mlp",
    "unpatch_qwen3_mlp",
]
