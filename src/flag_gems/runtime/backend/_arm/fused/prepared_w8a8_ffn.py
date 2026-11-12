# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared W8 FFN composition in one CPU Triton TLE Raw entry.

The original route composes separate native OpenMP regions. Experimental team
and typed routes can share a worker team while preserving BF16 rounding.
"""

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call

from .prepared_elementwise import PreparedGeluErfRaw, _entry as _elementwise_entry
from .prepared_w8a8_add_norm import PreparedW8A8AddNorm, _norm_entry
from .prepared_w8a8_sme import PreparedW8A8SME, _entry as _linear_entry, _scratch


@triton.jit(do_not_specialize_on_alignment=["descriptor"])
def _ffn_raw(descriptor, UP: tl.constexpr, DOWN: tl.constexpr):
    address = descriptor.to(tl.uint64)
    raw_call(UP, address, result_dtype=None)
    raw_call("flag_gems_arm_gelu_lookup_bf16", address + 17 * 8, result_dtype=None)
    raw_call(DOWN, address + 22 * 8, result_dtype=None)


@triton.jit(do_not_specialize_on_alignment=["descriptor"])
def _ffn_team_raw(descriptor, SYMBOL: tl.constexpr):
    raw_call(SYMBOL, descriptor.to(tl.uint64), result_dtype=None)


@lru_cache(maxsize=1)
def _tensor_ffn_entry():
    from ..tle_raw.native_raw import initialize_native_raw

    path = os.environ["FLAGGEMS_ARM_FFN_TEAM_LIBRARY"]
    initialize_native_raw(path)
    op = torch.ops.flag_gems_arm_w8a8_sme.ffn.raw

    @torch.library.register_fake("flag_gems_arm_w8a8_sme::ffn.raw")
    def fake(
        x,
        residual,
        uw,
        ui8,
        us,
        uinv,
        ub,
        dw,
        di8,
        ds,
        dinv,
        db,
        nw,
        nb,
        eps_bits,
        ut,
        ust,
        dt,
        dst,
        gt,
        grt,
        nt,
    ):
        return torch.empty((*x.shape[:-1], ds.numel()), dtype=x.dtype, device=x.device)

    return op


@lru_cache(maxsize=4)
def _entry(up_scaled, down_scaled):
    experiment = os.environ.get("FLAGGEMS_ARM_FFN_TEAM_ROUTE", "original")
    if experiment not in ("original", "composed", "team", "tensor"):
        raise ValueError("Unsupported experimental FFN route")
    if experiment == "tensor":
        return None, None, None, _tensor_ffn_entry()
    if experiment != "original":
        path = Path(os.environ["FLAGGEMS_ARM_FFN_TEAM_LIBRARY"])
        native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        symbol = "flag_gems_arm_w8a8_sme_ffn_" + experiment
        if not hasattr(native, symbol):
            raise RuntimeError(f"Missing experimental FFN symbol {symbol}")
        compiled = _ffn_team_raw.warmup(
            torch.empty(1, dtype=torch.uint8),
            SYMBOL=symbol,
            grid=(1,),
            num_cpu_threads=1,
        )
        library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
        function = getattr(library, compiled.name)
        function.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
        function.restype = None
        return native, compiled, library, function
    owners = (
        _linear_entry(up_scaled, hybrid=True),
        _norm_entry(down_scaled),
        _elementwise_entry("flag_gems_arm_gelu_lookup_bf16"),
    )
    up = "flag_gems_arm_w8a8_sme" + ("_scaled" if up_scaled else "") + "_hybrid"
    down = (
        "flag_gems_arm_w8a8_sme" + ("_scaled" if down_scaled else "") + "_hybrid_norm"
    )
    compiled = _ffn_raw.warmup(
        torch.empty(1, dtype=torch.uint8),
        UP=up,
        DOWN=down,
        grid=(1,),
        num_cpu_threads=1,
    )
    library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    function = getattr(library, compiled.name)
    function.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    function.restype = None
    return owners, compiled, library, function


class PreparedW8A8FFN:
    """W8 Linear -> exact BF16 erf-GELU -> W8 Linear -> residual/LayerNorm.

    The original descriptor route supports up to32 flattened rows; typed builds
    validate their separately qualified row limit in the native operator.
    The returned output is always fresh;
    the in-place GELU buffer is private scratch, never an exposed input/output.
    """

    def __init__(self, up, down_norm, gelu):
        if (
            not isinstance(up, PreparedW8A8SME)
            or up.kernel != "hybrid"
            or not isinstance(down_norm, PreparedW8A8AddNorm)
            or up.n != down_norm.linear.k
            or not isinstance(gelu, PreparedGeluErfRaw)
        ):
            raise ValueError(
                "Compatible hybrid W8 plans, residual norm and erf-GELU required"
            )
        self.up, self.down_norm, self.gelu = up, down_norm, gelu
        self.entry = _entry(
            up.inverse is not None, down_norm.linear.inverse is not None
        )
        self.tensor_op = (
            self.entry[3]
            if os.environ.get("FLAGGEMS_ARM_FFN_TEAM_ROUTE") == "tensor"
            else None
        )
        self.calls = 0

    def __call__(
        self,
        x,
        residual,
        bias_up=None,
        bias_down=None,
        *,
        up_threads=None,
        up_sme_threads=None,
        down_threads=None,
        down_sme_threads=None,
    ):
        up, norm, gelu = self.up, self.down_norm, self.gelu
        down = norm.linear
        if up.profile_enabled or down.profile_enabled:
            raise RuntimeError(
                "Use composed-operator timing; separate native-stage profiling is not supported"
            )
        if self.tensor_op is not None:
            output = self.tensor_op(
                x,
                residual,
                up.packed,
                up.packed_i8mm,
                up.scales,
                up.inverse,
                bias_up,
                down.packed,
                down.packed_i8mm,
                down.scales,
                down.inverse,
                bias_down,
                norm.weight,
                norm.bias,
                norm.eps_bits,
                up.threads if up_threads is None else up_threads,
                up.sme_threads if up_sme_threads is None else up_sme_threads,
                down.threads if down_threads is None else down_threads,
                down.sme_threads if down_sme_threads is None else down_sme_threads,
                gelu.threads,
                gelu.reference_threads or torch.get_num_threads(),
                norm.norm_threads,
            )
            if not torch.compiler.is_compiling() and x.numel():
                up.calls += 1
                down.calls += 1
                self.calls += 1
            return output
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or x.ndim < 2
            or x.shape[-1] != up.k
        ):
            raise ValueError("CPU BF16 FFN input ending in up K required")
        shape = (*x.shape[:-1], down.n)
        if (
            residual.device != x.device
            or residual.dtype != x.dtype
            or residual.shape != shape
            or not residual.is_contiguous()
        ):
            raise ValueError("Contiguous BF16 residual matching FFN output required")
        for bias, width in [(bias_up, up.n), (bias_down, down.n)]:
            if bias is not None and (
                bias.device != x.device
                or bias.dtype != x.dtype
                or bias.shape != (width,)
                or not bias.is_contiguous()
            ):
                raise ValueError("Contiguous CPU BF16 FFN bias required")
        inputs = [x, residual, norm.weight, norm.bias] + [
            b for b in [bias_up, bias_down] if b is not None
        ]
        if torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
            raise RuntimeError("FFN composition is inference only")
        ut = up.threads if up_threads is None else up_threads
        us = up.sme_threads if up_sme_threads is None else up_sme_threads
        dt = down.threads if down_threads is None else down_threads
        ds = down.sme_threads if down_sme_threads is None else down_sme_threads
        for threads, sme in [(ut, us), (dt, ds)]:
            if (
                type(threads) is not int
                or not 1 <= threads <= (os.cpu_count() or 1)
                or type(sme) is not int
                or not 0 <= sme <= threads
            ):
                raise ValueError("Invalid FFN worker configuration")
        flat = x.reshape(-1, up.k).contiguous()
        m = flat.shape[0]
        if m > 32:
            raise ValueError("FFN composition is qualified for at most32 rows")
        output = torch.empty((m, down.n), dtype=torch.bfloat16)
        if m:
            mp = (m + 15) // 16 * 16
            maximum_k = max(up.k, down.k)
            for key, size, dtype in [
                ("a", mp * maximum_k, torch.int8),
                ("ai", mp * maximum_k, torch.int8),
                ("scale", mp, torch.float32),
                ("ffn_up", m * up.n, torch.bfloat16),
            ]:
                value = getattr(_scratch, key, None)
                if value is None or value.numel() < size:
                    setattr(_scratch, key, torch.empty(size, dtype=dtype))
            a, ai, scale, hidden = (
                _scratch.a,
                _scratch.ai,
                _scratch.scale,
                _scratch.ffn_up,
            )

            def linear_descriptor(plan, source, target, bias, threads, sme):
                return [
                    source,
                    plan.packed.data_ptr(),
                    plan.scales.data_ptr(),
                    a.data_ptr(),
                    scale.data_ptr(),
                    0,
                    target,
                    0 if bias is None else bias.data_ptr(),
                    m,
                    plan.n,
                    plan.k,
                    threads,
                    0 if plan.inverse is None else plan.inverse.data_ptr(),
                    0,
                    plan.packed_i8mm.data_ptr(),
                    ai.data_ptr(),
                    sme,
                ]

            fields = linear_descriptor(
                up, flat.data_ptr(), hidden.data_ptr(), bias_up, ut, us
            )
            fields += [
                hidden.data_ptr(),
                hidden.data_ptr(),
                m * up.n,
                gelu.threads,
                gelu.reference_threads or torch.get_num_threads(),
            ]
            fields += linear_descriptor(
                down, hidden.data_ptr(), output.data_ptr(), bias_down, dt, ds
            )
            fields += [
                residual.data_ptr(),
                norm.weight.data_ptr(),
                norm.bias.data_ptr(),
                norm.eps_bits,
                norm.norm_threads,
            ]
            descriptor = (ctypes.c_int64 * 44)(*fields)
            self.entry[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
            up.calls += 1
            down.calls += 1
            self.calls += 1
        return output.reshape(shape)
