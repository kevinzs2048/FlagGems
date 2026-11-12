# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in prepared W8 linear, BF16 residual and LayerNorm CPU TLE Raw operator."""

import ctypes
from functools import lru_cache
import math
import os
import struct
import torch
from .prepared_w8a8_sme import _entry, _scratch, _w8a8_sme, _tensor_linear_entry


@lru_cache(maxsize=2)
def _norm_entry(scaled):
    native = _entry(scaled, False, False, True)[0]
    if os.environ.get("FLAGGEMS_ARM_W8_NORM_LIBRARY"):
        native = ctypes.CDLL(
            os.environ["FLAGGEMS_ARM_W8_NORM_LIBRARY"], mode=ctypes.RTLD_GLOBAL
        )
    if (
        not hasattr(native, "flag_gems_arm_w8a8_norm_abi_version")
        or native.flag_gems_arm_w8a8_norm_abi_version() != 2
    ):
        raise RuntimeError("Fused norm requires the worker-bounded ABI2 build")
    symbol = "flag_gems_arm_w8a8_sme" + ("_scaled" if scaled else "") + "_hybrid_norm"
    if not hasattr(native, symbol):
        raise RuntimeError(
            "Build the optional W8 residual norm leaf before selecting it"
        )
    compiled = _w8a8_sme.warmup(
        torch.empty(1, dtype=torch.uint8), SYMBOL=symbol, grid=(1,), num_cpu_threads=1
    )
    dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    fn = getattr(dso, compiled.name)
    fn.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    fn.restype = None
    return native, compiled, dso, fn


class PreparedW8A8AddNorm:
    def __init__(self, linear, weight, bias, eps=1e-5, *, norm_threads=4):
        if linear.kernel != "hybrid" or not 1 <= linear.n <= 4096:
            raise ValueError("Hybrid W8 plan with norm width at most4096 required")
        for tensor in [weight, bias]:
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.bfloat16
                or tensor.shape != (linear.n,)
                or not tensor.is_contiguous()
            ):
                raise ValueError("Contiguous CPU BF16 norm parameters required")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Positive finite epsilon required")
        if not isinstance(norm_threads, int) or not 1 <= norm_threads <= (
            os.cpu_count() or 1
        ):
            raise ValueError("Positive norm worker count required")
        self.norm_threads = norm_threads
        self.linear, self.weight, self.bias = linear, weight, bias
        self.eps_bits = struct.unpack("I", struct.pack("f", eps))[0]
        self.entry = _norm_entry(linear.inverse is not None)

    def __call__(self, x, residual, bias=None, *, threads=None, sme_threads=None):
        plan = self.linear
        if (
            os.environ.get("FLAGGEMS_ARM_SME_LINEAR_TENSOR_RAW") == "1"
            and not plan.profile_enabled
        ):
            if getattr(self, "tensor_op", None) is None:
                self.tensor_op = _tensor_linear_entry()
            result = self.tensor_op(
                x,
                plan.packed,
                plan.packed_i8mm,
                plan.scales,
                plan.inverse,
                bias,
                residual,
                self.weight,
                self.bias,
                self.eps_bits,
                plan.threads if threads is None else threads,
                plan.sme_threads if sme_threads is None else sme_threads,
                self.norm_threads,
            )
            if not torch.compiler.is_compiling() and x.numel():
                plan.calls += 1
            return result
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or x.ndim < 2
            or x.shape[-1] != plan.k
        ):
            raise ValueError("CPU BF16 input ending in linear K required")
        shape = (*x.shape[:-1], plan.n)
        if (
            residual.device != x.device
            or residual.dtype != x.dtype
            or residual.shape != shape
            or not residual.is_contiguous()
        ):
            raise ValueError("Contiguous BF16 residual matching output required")
        if bias is not None and (
            bias.device != x.device
            or bias.dtype != x.dtype
            or bias.shape != (plan.n,)
            or not bias.is_contiguous()
        ):
            raise ValueError("Contiguous BF16 linear bias required")
        if torch.is_grad_enabled() and any(
            t.requires_grad
            for t in [x, residual, self.weight, self.bias]
            + ([] if bias is None else [bias])
        ):
            raise RuntimeError("Inference only")
        threads = plan.threads if threads is None else threads
        sme_threads = plan.sme_threads if sme_threads is None else sme_threads
        if (
            not isinstance(threads, int)
            or not 1 <= threads <= (os.cpu_count() or 1)
            or not isinstance(sme_threads, int)
            or not 0 <= sme_threads <= threads
        ):
            raise ValueError("Invalid worker configuration")
        flat = x.reshape(-1, plan.k).contiguous()
        m = flat.shape[0]
        mp = (m + 15) // 16 * 16
        output = torch.empty((m, plan.n), dtype=torch.bfloat16)
        if m:
            for name, size, dtype in [
                ("a", mp * plan.k, torch.int8),
                ("ai", mp * plan.k, torch.int8),
                ("scale", mp, torch.float32),
            ]:
                value = getattr(_scratch, name, None)
                if value is None or value.numel() < size:
                    setattr(_scratch, name, torch.empty(size, dtype=dtype))
            desc = (ctypes.c_int64 * 22)(
                flat.data_ptr(),
                plan.packed.data_ptr(),
                plan.scales.data_ptr(),
                _scratch.a.data_ptr(),
                _scratch.scale.data_ptr(),
                0,
                output.data_ptr(),
                0 if bias is None else bias.data_ptr(),
                m,
                plan.n,
                plan.k,
                threads,
                0 if plan.inverse is None else plan.inverse.data_ptr(),
                0,
                plan.packed_i8mm.data_ptr(),
                _scratch.ai.data_ptr(),
                sme_threads,
                residual.data_ptr(),
                self.weight.data_ptr(),
                self.bias.data_ptr(),
                self.eps_bits,
                self.norm_threads,
            )
            self.entry[3](ctypes.addressof(desc), 0, 0, 0, 1, 1, 1)
            plan.calls += 1
        return output.reshape(shape)
