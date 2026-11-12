# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared BF16 linear through CPU Triton TLE Raw and native SME/SME2.

Inference only. BF16 operands are preserved; accumulation is FP32. This is
an explicit Apple SME2 route, never a silent fallback or a quantized GEMM.
Prepared weights must be recreated after mutation (including inference tensors,
which have no PyTorch version counter). Scratch storage is per calling thread
and shared across plans; calls through the CPU entry are synchronous.
"""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import platform
import subprocess
import threading

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call


@triton.jit(do_not_specialize_on_alignment=["descriptor_ptr"])
def _bf16_sme_raw_kernel(descriptor_ptr, VARIANT: tl.constexpr):
    if VARIANT == 0:
        raw_call(
            "flag_gems_arm_bf16_sme", descriptor_ptr.to(tl.uint64), result_dtype=None
        )
    elif VARIANT == 1:
        raw_call(
            "flag_gems_arm_bf16_sme_16x64",
            descriptor_ptr.to(tl.uint64),
            result_dtype=None,
        )
    elif VARIANT == 2:
        raw_call(
            "flag_gems_arm_bf16_sme_f32_16x64",
            descriptor_ptr.to(tl.uint64),
            result_dtype=None,
        )
    elif VARIANT == 3:
        raw_call(
            "flag_gems_arm_bf16_sme_16x64_striped",
            descriptor_ptr.to(tl.uint64),
            result_dtype=None,
        )
    else:
        raw_call(
            "flag_gems_arm_bf16_sme_torch_16x64",
            descriptor_ptr.to(tl.uint64),
            result_dtype=None,
        )


@lru_cache(maxsize=5)
def _entry(variant=0):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("BF16 SME route currently requires Apple arm64 SME2")
    feature = subprocess.run(
        ["sysctl", "-n", "hw.optional.arm.FEAT_SME2"],
        capture_output=True,
        text=True,
        check=False,
    )
    if feature.returncode or feature.stdout.strip() != "1":
        raise RuntimeError("BF16 SME route requires SME2")
    default = (
        Path(__file__).resolve().parents[1] / "lib/libflag_gems_arm_bf16_sme.dylib"
    )
    path = Path(os.getenv("FLAGGEMS_ARM_BF16_SME_LIBRARY", str(default))).resolve()
    if not path.is_file():
        raise RuntimeError("Build the flag_gems_arm_bf16_sme CMake target first")
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    width = native.flag_gems_arm_bf16_sme_vector_words
    width.argtypes = []
    width.restype = ctypes.c_int
    if width() != 16:
        raise RuntimeError("BF16 SME kernel requires 512-bit streaming vector length")
    compiled = _bf16_sme_raw_kernel.warmup(
        torch.empty(1, dtype=torch.uint8, device="cpu"),
        VARIANT=variant,
        grid=(1,),
        num_cpu_threads=1,
    )
    dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    function = getattr(dso, compiled.name)
    function.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    function.restype = None
    return native, compiled, dso, function


_scratch = threading.local()


def _workspace(a_elements, c_elements, a_dtype):
    a_name = "a_f32" if a_dtype == torch.float32 else "a_bf16"
    for name, count, dtype in (
        (a_name, a_elements, a_dtype),
        ("c", c_elements, torch.float32),
    ):
        current = getattr(_scratch, name, None)
        if current is None or current.dtype != dtype or current.numel() < count:
            setattr(_scratch, name, torch.empty(count, dtype=dtype, device="cpu"))
    return getattr(_scratch, a_name), _scratch.c


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # inference tensors have no version counter
        return None


class PreparedBF16SME:
    """An immutable packed weight owner; use for multi-token inference linears."""

    def __init__(self, weight, threads=1, kernel="bf16_32x32"):
        if (
            weight.device.type != "cpu"
            or weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or min(weight.shape) < 1
        ):
            raise ValueError("A nonempty CPU BF16 [N,K] weight is required")
        if not isinstance(threads, int) or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("threads must be between 1 and the host CPU count")
        variants = {
            "bf16_32x32": 0,
            "bf16_16x64": 1,
            "fp32_16x64": 2,
            "bf16_16x64_striped": 3,
            "fp32_torch_16x64": 4,
        }
        if kernel not in variants:
            raise ValueError(f"Unknown SME kernel: {kernel}")
        self.variant = variants[kernel]
        if self.variant == 4 and not weight.is_contiguous():
            raise ValueError(
                "Torch-order SME requires a contiguous weight; dispatch other layouts explicitly"
            )
        self.mr, self.nr = (32, 32) if self.variant == 0 else (16, 64)
        self.packed_dtype = torch.float32 if self.variant in (2, 4) else torch.bfloat16
        _entry(self.variant)  # fail before allocating a large packed copy
        self.weight = weight
        self.version = _version(weight)
        self.n, self.k = weight.shape
        self.np = (self.n + self.nr - 1) // self.nr * self.nr
        self.kp = (
            (self.k + 31) // 32 * 32
            if self.variant == 4
            else self.k if self.variant == 2 else (self.k + 1) // 2 * 2
        )
        self.threads = threads
        self.calls = 0
        with torch.no_grad():
            padded = F.pad(
                weight.detach().to(self.packed_dtype),
                (0, self.kp - self.k, 0, self.np - self.n),
            )
            if self.variant == 4:
                self.packed = (
                    padded.reshape(self.np // self.nr, self.nr, self.kp // 32, 32)
                    .permute(0, 3, 2, 1)
                    .contiguous()
                )
            elif self.variant == 2:
                self.packed = (
                    padded.reshape(self.np // self.nr, self.nr, self.kp)
                    .permute(0, 2, 1)
                    .contiguous()
                )
            else:
                self.packed = (
                    padded.reshape(self.np // self.nr, self.nr, self.kp // 2, 2)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

    def __call__(self, x, bias=None):
        if torch.is_grad_enabled() and (
            x.requires_grad
            or self.weight.requires_grad
            or bias is not None
            and bias.requires_grad
        ):
            raise RuntimeError("PreparedBF16SME is an inference-only operator")
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or x.ndim < 1
            or x.shape[-1] != self.k
        ):
            raise ValueError("Expected a CPU BF16 input ending in K")
        if _version(self.weight) != self.version:
            raise RuntimeError("Weight changed; recreate PreparedBF16SME")
        if bias is not None and (
            bias.device.type != "cpu"
            or bias.dtype != torch.bfloat16
            or bias.shape != (self.n,)
            or not bias.is_contiguous()
        ):
            raise ValueError("Expected a contiguous CPU BF16 bias [N]")
        if self.variant == 4 and not x.is_contiguous():
            raise ValueError(
                "Torch-order SME requires contiguous input; PyTorch can use a different reduction for strided input"
            )
        flat = x.reshape(-1, self.k).contiguous()
        m = flat.shape[0]
        output = torch.empty((m, self.n), dtype=torch.bfloat16, device="cpu")
        if m:
            mp = (m + self.mr - 1) // self.mr * self.mr
            packed_a, scratch = _workspace(
                mp * self.kp, mp * self.np, self.packed_dtype
            )
            descriptor = torch.tensor(
                [
                    flat.data_ptr(),
                    self.packed.data_ptr(),
                    packed_a.data_ptr(),
                    scratch.data_ptr(),
                    output.data_ptr(),
                    m,
                    self.n,
                    self.k,
                    self.threads,
                    0 if bias is None else bias.data_ptr(),
                ],
                dtype=torch.int64,
                device="cpu",
            )
            _entry(self.variant)[3](descriptor.data_ptr(), 0, 0, 0, 1, 1, 1)
            self.calls += 1
        return output.reshape(*x.shape[:-1], self.n)

    def linear_with_shape_policy(self, x, bias=None):
        """Conservative inference dispatch for the OCR candidate profile.

        Accelerate multi-token expansion projections only. Keep single-token
        and contraction projections on the original BF16 framework route:
        changing their reduction order can alter autoregressive output. This
        geometry policy still requires model-level qualification for new models.
        """
        m = x.numel() // x.shape[-1]
        if (
            m >= 32
            and self.n >= 2 * self.k
            and (self.variant != 4 or x.is_contiguous())
        ):
            return self(x, bias)
        return F.linear(x, self.weight, bias)
