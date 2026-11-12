# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Owned I8MM weight packing and synchronous CPU Triton TLE Raw entries."""

import ctypes
from functools import lru_cache
import os
import platform
from pathlib import Path
import subprocess
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call


@triton.jit(do_not_specialize_on_alignment=["descriptor"])
def _i8mm_raw(descriptor, SYMBOL: tl.constexpr):
    raw_call(SYMBOL, descriptor.to(tl.uint64), result_dtype=None)


@lru_cache(maxsize=1)
def _require_i8mm():
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        supported = False
    elif sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.optional.arm.FEAT_I8MM"],
            capture_output=True,
            text=True,
        )
        supported = result.returncode == 0 and result.stdout.strip() == "1"
    elif sys.platform == "linux":
        libc = ctypes.CDLL(None)
        libc.getauxval.argtypes = [ctypes.c_ulong]
        libc.getauxval.restype = ctypes.c_ulong
        supported = bool(libc.getauxval(26) & (1 << 13))  # AT_HWCAP2 / HWCAP2_I8MM
    else:
        supported = False
    if not supported:
        raise RuntimeError("Explicit I8MM backend requires Arm FEAT_I8MM")


@lru_cache(maxsize=3)
def _entry(symbol):
    _require_i8mm()
    suffix = "dylib" if sys.platform == "darwin" else "so"
    default = (
        Path(__file__).resolve().parents[1]
        / "lib"
        / f"libflag_gems_arm_w8g32_i8mm.{suffix}"
    )
    path = Path(os.getenv("FLAGGEMS_ARM_W8G32_I8MM_LIBRARY", str(default))).resolve()
    if not path.is_file():
        raise RuntimeError(
            "Build the flag_gems_arm_w8g32_i8mm target before selecting I8MM"
        )
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    compiled = _i8mm_raw.warmup(
        torch.empty(1, dtype=torch.uint8, device="cpu"),
        SYMBOL=symbol,
        grid=(1,),
        num_cpu_threads=1,
    )
    dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    function = getattr(dso, compiled.name)
    # The locked CPU Triton ABI, also used by the SDOT route.
    function.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    function.restype = None
    return native, compiled, dso, function


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors have no counter. As for other prepacked formats,
        # callers must explicitly prepare again after updating model weights.
        return None


class PreparedI8MM:
    """Own source tensors and their derived layouts for the model lifetime."""

    def __init__(self, weight, scales, packed, packed_scales):
        self.n, self.k = weight.shape
        self.weight = weight
        self.scales = scales
        self.packed = packed
        self.packed_scales = packed_scales
        self.versions = (_version(weight), _version(scales))

    def validate(self, weight, scales):
        if (
            weight is not self.weight
            or scales is not self.scales
            or self.versions != (_version(weight), _version(scales))
        ):
            raise ValueError(
                "I8MM prepared weights must be rebuilt after changing source tensors"
            )


def prepare_i8mm(weight, scales, threads=None):
    _require_i8mm()
    if (
        weight.device.type != "cpu"
        or scales.device.type != "cpu"
        or weight.dtype != torch.int8
        or scales.dtype != torch.float32
    ):
        raise ValueError("I8MM packing requires CPU INT8 weights and FP32 scales")
    if weight.ndim != 2 or not weight.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Contiguous weights/scales required")
    n, k = weight.shape
    if k % 32 or tuple(scales.shape) != (n, k // 32):
        raise ValueError("G32 shape mismatch")
    threads = int(
        threads
        if threads is not None
        else os.getenv("OMP_NUM_THREADS", str(torch.get_num_threads()))
    )
    if threads < 1:
        raise ValueError("Positive thread count required")
    packed = torch.empty(
        (triton.cdiv(n, 8), k // 32, 4, 64), dtype=torch.int8, device="cpu"
    )
    packed_scales = torch.empty(
        (triton.cdiv(n, 8), k // 32, 8), dtype=torch.float32, device="cpu"
    )
    descriptor = torch.tensor(
        [
            weight.data_ptr(),
            scales.data_ptr(),
            packed.data_ptr(),
            packed_scales.data_ptr(),
            n,
            k,
            threads,
        ],
        dtype=torch.int64,
        device="cpu",
    )
    if n and k:
        _entry("flag_gems_arm_w8g32_i8mm_pack")[3](
            descriptor.data_ptr(), 0, 0, 0, 1, 1, 1
        )
    return PreparedI8MM(weight, scales, packed, packed_scales)


def run_i8mm(
    q, activation_scales, prepared, output, threads, grouped_input, per_channel=False
):
    if grouped_input and per_channel:
        raise ValueError("Per-channel accumulation requires per-token A8")
    m, k = q.shape
    n = prepared.n
    descriptor = torch.tensor(
        [
            q.data_ptr(),
            activation_scales.data_ptr(),
            prepared.packed.data_ptr(),
            prepared.packed_scales.data_ptr(),
            output.data_ptr(),
            m,
            n,
            k,
            min(threads, triton.cdiv(m, 4) * triton.cdiv(n, 8)),
            2 if per_channel else int(grouped_input),
        ],
        dtype=torch.int64,
        device="cpu",
    )
    _entry("flag_gems_arm_w8g32_i8mm")[3](descriptor.data_ptr(), 0, 0, 0, 1, 1, 1)
