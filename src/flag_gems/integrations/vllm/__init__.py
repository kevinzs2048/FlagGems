"""Public vLLM integration entry points."""

from .coverage import maybe_install_kernel_coverage
from .qwen_gdn import install_vllm_gdn as install_qwen_gdn

__all__ = ["install_qwen_gdn", "maybe_install_kernel_coverage"]
