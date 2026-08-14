"""Resolve the packaged Arm Q4/W8/GDN operator bundle."""

from __future__ import annotations

import os
from pathlib import Path


_ROOT = Path(__file__).resolve().parent


def configure_kernel_sources() -> None:
    """Point libtriton_jit at relocatable Python entry modules."""
    os.environ.setdefault(
        "FLAGGEMS_Q4_KERNEL_SOURCE", str(_ROOT / "q4_kernels.py")
    )
    os.environ.setdefault(
        "FLAGGEMS_W8_KERNEL_SOURCE", str(_ROOT / "w8_kernels.py")
    )


def library_path() -> Path:
    """Return the packaged native library or raise a precise error."""
    configured = os.getenv("FLAGGEMS_LIBTRITON_JIT_Q4_OP")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"configured FlagGems Arm library is missing: {path}")

    candidates = (
        "libflag_gems_arm_ops.dylib",
        "libflag_gems_arm_ops.so",
        "flag_gems_arm_ops.dll",
    )
    for name in candidates:
        path = _ROOT / name
        if path.is_file():
            return path
    expected = ", ".join(str(_ROOT / name) for name in candidates)
    raise FileNotFoundError(f"FlagGems Arm operator library not found; tried: {expected}")


def configure_runtime() -> Path:
    """Configure source paths and return the native operator library."""
    configure_kernel_sources()
    path = library_path()
    os.environ.setdefault("FLAGGEMS_LIBTRITON_JIT_Q4_OP", str(path))
    return path


__all__ = ["configure_kernel_sources", "configure_runtime", "library_path"]
