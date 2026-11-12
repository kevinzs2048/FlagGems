# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Noncausal BF16 SME attention through a cached CPU TLE Raw entry."""

import ctypes
from functools import lru_cache
from pathlib import Path
import os
import platform
import struct
import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call


@triton.jit(do_not_specialize_on_alignment=["descriptor"])
def _vision_attention_sme_raw_kernel(descriptor):
    raw_call(
        "flag_gems_arm_bf16_attention", descriptor.to(tl.uint64), result_dtype=None
    )


@lru_cache(maxsize=1)
def _vision_attention_sme_entry():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "Vision SME2 attention is currently qualified on Apple arm64"
        )
    from .bf16_sme import _entry

    _entry()  # Qualified SME2 hardware and 512-bit streaming vector length.
    default = (
        Path(__file__).resolve().parents[1]
        / "lib/libflag_gems_arm_bf16_attention_sme.dylib"
    )
    path = Path(os.getenv("FLAGGEMS_ARM_ATTENTION_SME_LIBRARY", str(default))).resolve()
    if not path.is_file():
        raise RuntimeError("Build flag_gems_arm_bf16_attention_sme first")
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    direct = native.flag_gems_arm_bf16_attention
    direct.argtypes = [ctypes.c_uint64]
    direct.restype = None
    compiled = _vision_attention_sme_raw_kernel.warmup(
        torch.empty(1, dtype=torch.uint8), grid=(1,), num_cpu_threads=1
    )
    owner = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    function = getattr(owner, compiled.name)
    function.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    function.restype = None
    register = native.flag_gems_arm_bf16_attention_register_raw
    register.argtypes = [ctypes.c_uint64]
    register.restype = None
    register(ctypes.cast(function, ctypes.c_void_p).value)
    launcher = native.flag_gems_arm_bf16_attention_raw
    launcher.argtypes = [ctypes.c_uint64]
    launcher.restype = None
    return native, compiled, owner, launcher, direct


def vision_attention_sme_bf16(q, k, v, *, scale=None, threads=4, backend="raw"):
    """Noncausal BF16 [B,H,S,D], D=64, through CPU TLE Raw.

    The native SME2 leaf preserves the qualified Torch CPU four-way GEMM
    sums, BF16 softmax weights and FP32 online state. The native entry is
    exposed only for explicit paired kernel benchmarks; no silent fallback.
    """
    if (
        q.ndim != 4
        or min(q.shape) < 1
        or q.shape[-1] != 64
        or any(
            x.shape != q.shape
            or x.dtype != torch.bfloat16
            or x.device.type != "cpu"
            or x.stride(-1) != 1
            for x in (q, k, v)
        )
    ):
        raise ValueError(
            "Expected equal CPU BF16 Q/K/V [B,H,S,D], D=64, with contiguous head dimension"
        )
    if not isinstance(threads, int) or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("Invalid CPU thread count")
    if backend not in ("raw", "native"):
        raise ValueError("backend must be raw or native")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise RuntimeError("Vision SME2 attention is inference-only")
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    scale = float(q.shape[-1] ** -0.5 if scale is None else scale)
    descriptor = torch.tensor(
        [
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            output.data_ptr(),
            q.shape[0],
            q.shape[1],
            k.shape[1],
            q.shape[2],
            k.shape[2],
            q.shape[3],
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            threads,
            0,
            struct.unpack("I", struct.pack("f", scale))[0],
        ],
        dtype=torch.int64,
    )
    entry = _vision_attention_sme_entry()
    if backend == "raw":
        entry[3](descriptor.data_ptr())
    else:
        entry[4](descriptor.data_ptr())
    return output
