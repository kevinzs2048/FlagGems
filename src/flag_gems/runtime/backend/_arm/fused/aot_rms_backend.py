"""Optional direct loader for shape-specialized ordinary-Triton RMSNorm."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch


class AOTRMSBackend:
    def __init__(
        self,
        library: Path,
        shape_dir: Path,
        rows: int,
        cols: int,
        fused_add: bool = False,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self._handle = None
        self._lib = ctypes.CDLL(str(library))
        create = (
            self._lib.triton_bf16_fused_add_rms_kernel_create
            if fused_add
            else self._lib.triton_bf16_rms_kernel_create
        )
        create.argtypes = [ctypes.c_char_p, ctypes.c_int64]
        create.restype = ctypes.c_void_p
        self._lib.triton_bf16_rms_kernel_destroy.argtypes = [
            ctypes.c_void_p
        ]
        self._launch = (
            self._lib.triton_bf16_fused_add_rms_launch
            if fused_add
            else self._lib.triton_bf16_rms_launch
        )
        self._launch.argtypes = [ctypes.c_void_p] * 4
        self._launch.restype = ctypes.c_int
        self._lib.triton_bf16_w8_last_error.restype = ctypes.c_char_p
        self._handle = create(str(shape_dir).encode(), rows)
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        message = self._lib.triton_bf16_w8_last_error()
        return message.decode() if message else "unknown BF16 RMSNorm AOT error"

    def close(self) -> None:
        if self._handle:
            self._lib.triton_bf16_rms_kernel_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        status = self._launch(
            self._handle,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(self._error())


def create_aot_rms_backend(
    rows: int, cols: int, eps: float
) -> AOTRMSBackend | None:
    enabled = os.getenv("FLAGGEMS_ARM_RMS_AOT", "1").lower()
    if enabled not in {"1", "true", "on"}:
        return None
    bundle_value = os.getenv("FLAGGEMS_ARM_W8_AOT_BUNDLE", "")
    if not bundle_value or eps != 1.0e-6:
        return None
    bundle = Path(bundle_value).resolve()
    shape_dir = bundle / f"rms-m{rows}-n{cols}-e1e-6"
    if not (shape_dir / "_rms_norm_aot_kernel.so").is_file():
        return None
    library_value = os.getenv("FLAGGEMS_ARM_W8_AOT_LIBRARY", "")
    library = (
        Path(library_value).resolve()
        if library_value
        else bundle.parent / "libtriton_bf16_w8_backend.so"
    )
    if not library.is_file():
        raise RuntimeError(f"BF16 RMSNorm AOT library not found: {library}")
    return AOTRMSBackend(library, shape_dir, rows, cols)


def create_aot_fused_add_rms_backend(
    rows: int, cols: int, eps: float
) -> AOTRMSBackend | None:
    enabled = os.getenv("FLAGGEMS_ARM_FUSED_RMS_AOT", "1").lower()
    if enabled not in {"1", "true", "on"}:
        return None
    bundle_value = os.getenv("FLAGGEMS_ARM_W8_AOT_BUNDLE", "")
    if not bundle_value or eps != 1.0e-6:
        return None
    bundle = Path(bundle_value).resolve()
    shape_dir = bundle / f"fused-rms-m{rows}-n{cols}-e1e-6"
    if not (shape_dir / "_fused_add_rms_aot_kernel.so").is_file():
        return None
    library_value = os.getenv("FLAGGEMS_ARM_W8_AOT_LIBRARY", "")
    library = (
        Path(library_value).resolve()
        if library_value
        else bundle.parent / "libtriton_bf16_w8_backend.so"
    )
    if not library.is_file():
        raise RuntimeError(f"BF16 fused RMSNorm AOT library not found: {library}")
    return AOTRMSBackend(library, shape_dir, rows, cols, fused_add=True)
