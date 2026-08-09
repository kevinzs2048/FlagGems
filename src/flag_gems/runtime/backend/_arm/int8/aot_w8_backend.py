"""Optional one-call loader for Triton-CPU 3.7 ordinary-dot W8 AOT pairs."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch


class AOTW8Backend:
    def __init__(
        self,
        library: Path,
        shape_dir: Path,
        k: int,
        n: int,
        block_n: int,
    ) -> None:
        self.k = k
        self.n = n
        self.block_n = block_n
        self._handle = None
        self._lib = ctypes.CDLL(str(library))
        create = self._lib.triton_bf16_w8_kernel_create_wide
        create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        create.restype = ctypes.c_void_p
        self._lib.triton_bf16_w8_kernel_destroy.argtypes = [ctypes.c_void_p]
        self._lib.triton_bf16_w8_launch.argtypes = [ctypes.c_void_p] * 5
        self._lib.triton_bf16_w8_launch.restype = ctypes.c_int
        self._lib.triton_bf16_w8_last_error.restype = ctypes.c_char_p
        self._handle = create(
            str(shape_dir).encode(),
            str(shape_dir).encode(),
            k,
            n,
            block_n,
        )
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        message = self._lib.triton_bf16_w8_last_error()
        return message.decode() if message else "unknown BF16-W8 AOT error"

    def close(self) -> None:
        if self._handle:
            self._lib.triton_bf16_w8_kernel_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __call__(
        self,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        status = self._lib.triton_bf16_w8_launch(
            self._handle,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed_weight.data_ptr()),
            ctypes.c_void_p(weight_scale.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(self._error())


class AOTW8MLPBackend:
    def __init__(
        self,
        library: Path,
        shape_dir: Path,
        k: int,
        n: int,
        block_n: int,
    ) -> None:
        self.k = k
        self.n = n
        self.block_n = block_n
        self._handle = None
        self._lib = ctypes.CDLL(str(library))
        create = self._lib.triton_bf16_w8_mlp_kernel_create
        create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        create.restype = ctypes.c_void_p
        self._lib.triton_bf16_w8_mlp_kernel_destroy.argtypes = [
            ctypes.c_void_p
        ]
        self._lib.triton_bf16_w8_mlp_launch.argtypes = [
            ctypes.c_void_p
        ] * 5
        self._lib.triton_bf16_w8_mlp_launch.restype = ctypes.c_int
        self._lib.triton_bf16_w8_last_error.restype = ctypes.c_char_p
        encoded = str(shape_dir).encode()
        self._handle = create(encoded, encoded, encoded, k, n, block_n)
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        message = self._lib.triton_bf16_w8_last_error()
        return message.decode() if message else "unknown BF16-W8 MLP AOT error"

    def close(self) -> None:
        if self._handle:
            self._lib.triton_bf16_w8_mlp_kernel_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __call__(
        self,
        x: torch.Tensor,
        packed_gate_up: torch.Tensor,
        gate_up_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        status = self._lib.triton_bf16_w8_mlp_launch(
            self._handle,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed_gate_up.data_ptr()),
            ctypes.c_void_p(gate_up_scale.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(self._error())


def create_aot_w8_backend(
    k: int, n: int, block_n: int = 64
) -> AOTW8Backend | None:
    bundle_value = os.getenv("FLAGGEMS_ARM_W8_AOT_BUNDLE", "")
    if not bundle_value:
        return None
    bundle = Path(bundle_value).resolve()
    shape_dir = bundle / f"k{k}-n{n}-bn{block_n}"
    if not (
        (shape_dir / "_quantize_bf16_w8_kernel.so").is_file()
        and (shape_dir / "_w8a8_wide_gemv_kernel.so").is_file()
    ):
        return None
    library_value = os.getenv("FLAGGEMS_ARM_W8_AOT_LIBRARY", "")
    library = (
        Path(library_value).resolve()
        if library_value
        else bundle.parent / "libtriton_bf16_w8_backend.so"
    )
    if not library.is_file():
        raise RuntimeError(f"BF16-W8 AOT backend library not found: {library}")
    return AOTW8Backend(library, shape_dir, k, n, block_n)


def create_aot_w8_mlp_backend(
    k: int, n: int, block_n: int = 64
) -> AOTW8MLPBackend | None:
    enabled = os.getenv("FLAGGEMS_ARM_W8_MLP_AOT", "1").lower()
    if enabled not in {"1", "true", "on"}:
        return None
    bundle_value = os.getenv("FLAGGEMS_ARM_W8_AOT_BUNDLE", "")
    if not bundle_value:
        return None
    bundle = Path(bundle_value).resolve()
    shape_dir = bundle / f"mlp-k{k}-n{n}-bn{block_n}"
    names = (
        "_quantize_bf16_w8_kernel.so",
        "_w8a8_wide_gemv_kernel.so",
        "_bf16_swiglu_kernel.so",
    )
    if not all((shape_dir / name).is_file() for name in names):
        return None
    library_value = os.getenv("FLAGGEMS_ARM_W8_AOT_LIBRARY", "")
    library = (
        Path(library_value).resolve()
        if library_value
        else bundle.parent / "libtriton_bf16_w8_backend.so"
    )
    if not library.is_file():
        raise RuntimeError(f"BF16-W8 AOT backend library not found: {library}")
    return AOTW8MLPBackend(library, shape_dir, k, n, block_n)
