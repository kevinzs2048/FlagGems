# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse the owned I8MM TLE Raw layout for per-channel W8A8 on Arm CPUs."""

import os
import ctypes
import threading
from functools import lru_cache
import torch
import triton
from .w8a8_group import _quantize_token
from .w8a8_group_i8mm import prepare_i8mm, run_i8mm, _entry

_scratch = threading.local()


@lru_cache(maxsize=1)
def _tensor_raw_operators(library_path):
    from flag_gems.runtime.backend._arm.tle_raw.native_raw import initialize_native_raw

    initialize_native_raw(library_path)

    @torch.library.register_fake("flag_gems_arm_w8a8_i8mm::linear.raw")
    def linear_meta(x, packed, scales, inverse, bias, threads, n_major):
        return torch.empty(
            (*x.shape[:-1], scales.numel()), dtype=x.dtype, device=x.device
        )

    @torch.library.register_fake("flag_gems_arm_w8a8_i8mm::add_norm.raw")
    def norm_meta(
        x,
        residual,
        packed,
        scales,
        inverse,
        bias,
        nw,
        nb,
        eps_bits,
        threads,
        norm_threads,
        n_major,
    ):
        return torch.empty_like(residual)

    return (
        torch.ops.flag_gems_arm_w8a8_i8mm.linear.raw,
        torch.ops.flag_gems_arm_w8a8_i8mm.add_norm.raw,
        linear_meta,
        norm_meta,
    )


class PreparedW8A8I8MM:
    def __init__(
        self, weight, threads=8, smoothing=None, *, kernel="staged", order="n"
    ):
        if (
            weight.device.type != "cpu"
            or weight.ndim != 2
            or not weight.is_floating_point()
            or min(weight.shape) < 1
            or weight.shape[1] % 32
        ):
            raise ValueError("CPU floating weight [N,K], K divisible by32 required")
        if not isinstance(threads, int) or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Invalid threads")
        self.n, self.k = weight.shape
        self.threads = threads
        self.calls = 0
        self.inverse = None
        self._tensor_raw_ops = None
        self.kernel = kernel
        if self.kernel not in {"staged", "fused"}:
            raise ValueError("Unknown I8MM kernel")
        if order not in {"n", "m", "auto"}:
            raise ValueError("I8MM order must be n, m or auto")
        if order == "auto" and kernel != "fused":
            raise ValueError("Automatic I8MM order requires the fused kernel")
        self.n_major = order == "n"
        self.auto_order = order == "auto"
        self.symbol = "flag_gems_arm_w8a8_i8mm_fused"
        native = _entry("flag_gems_arm_w8g32_i8mm")[0]
        if (
            not hasattr(native, "flag_gems_arm_w8a8_i8mm_abi_version")
            or native.flag_gems_arm_w8a8_i8mm_abi_version() != 2
        ):
            raise RuntimeError(
                "Per-channel W8A8 requires the FlagGems I8MM ABI2 library"
            )
        values = weight.detach().float()
        if smoothing is not None:
            if (
                smoothing.device.type != "cpu"
                or smoothing.shape != (self.k,)
                or not torch.isfinite(smoothing).all()
                or (smoothing <= 0).any()
            ):
                raise ValueError("Invalid smoothing")
            values = values * smoothing.float()[None, :]
            self.inverse = smoothing.float().reciprocal().contiguous()
        if not torch.isfinite(values).all():
            raise ValueError("Finite weights required")
        scales = (values.abs().amax(1) / 127).clamp_min(1e-8)
        quant = (
            (values / scales[:, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
            .contiguous()
        )
        if self.kernel != "staged":
            import torch.nn.functional as F

            padded = F.pad(quant, (0, 0, 0, (-self.n) % 8))
            self.packed = (
                padded.reshape(-1, 8, self.k // 8, 8).permute(0, 2, 1, 3).contiguous()
            )
            self.scales = scales.contiguous()
            _entry(self.symbol)
            if os.environ.get("FLAGGEMS_ARM_I8MM_TENSOR_RAW", "0") == "1":
                self._tensor_raw_ops = _tensor_raw_operators(native._name)
            return
        expanded = scales[:, None].expand(self.n, self.k // 32).contiguous()
        prepared = prepare_i8mm(quant, expanded, threads)
        self.packed = prepared.packed
        self.packed_scales = prepared.packed_scales
        self.scales = self.packed_scales
        # The packed ABI owns everything needed for execution; source copies are released.

    def _order_for(self, x):
        if self.auto_order and x.ndim and x.shape[-1] > 0:
            rows = x.numel() // x.shape[-1]
            return 64 < rows <= 128
        return self.n_major

    def __call__(self, x, bias=None):
        if self._tensor_raw_ops is not None:
            output = self._tensor_raw_ops[0](
                x,
                self.packed,
                self.scales,
                self.inverse,
                bias,
                self.threads,
                self._order_for(x),
            )
            if not torch.compiler.is_compiling():
                self.calls += 1
            return output
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or x.ndim < 1
            or x.shape[-1] != self.k
        ):
            raise ValueError("CPU BF16 activations ending in K required")
        if torch.is_grad_enabled() and (
            x.requires_grad or bias is not None and bias.requires_grad
        ):
            raise RuntimeError("Inference only")
        if bias is not None and (
            bias.device != x.device
            or bias.dtype != x.dtype
            or bias.shape != (self.n,)
            or not bias.is_contiguous()
        ):
            raise ValueError("Contiguous BF16 bias [N] required")
        flat = x.reshape(-1, self.k).contiguous()
        m = flat.shape[0]
        if not m:
            return x.new_empty((*x.shape[:-1], self.n))
        if self.kernel != "staged":
            size = triton.cdiv(m, 8) * 8
            for name, count, dtype in [
                ("a", size * self.k, torch.int8),
                ("scale", size, torch.float32),
            ]:
                buf = getattr(_scratch, name, None)
                if buf is None or buf.numel() < count:
                    setattr(_scratch, name, torch.empty(count, dtype=dtype))
            output = torch.empty((m, self.n), dtype=torch.bfloat16)
            descriptor = (ctypes.c_int64 * 13)(
                flat.data_ptr(),
                self.packed.data_ptr(),
                self.scales.data_ptr(),
                _scratch.a.data_ptr(),
                _scratch.scale.data_ptr(),
                output.data_ptr(),
                0 if bias is None else bias.data_ptr(),
                m,
                self.n,
                self.k,
                self.threads,
                0 if self.inverse is None else self.inverse.data_ptr(),
                int(self._order_for(flat)),
            )
            _entry(self.symbol)[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
            self.calls += 1
            return output.reshape(*x.shape[:-1], self.n)
        if self.inverse is not None:
            flat = flat.float() * self.inverse
        q = torch.empty((m, self.k), dtype=torch.int8)
        xs = torch.empty(m, dtype=torch.float32)
        output = torch.empty((m, self.n), dtype=torch.float32)
        _quantize_token[(m,)](
            flat,
            q,
            xs,
            xs,
            self.k,
            triton.next_power_of_2(self.k),
            True,
            num_cpu_threads=min(self.threads, m),
            enable_fp_fusion=False,
        )
        run_i8mm(q, xs, self, output, self.threads, False, per_channel=True)
        if bias is not None:
            output += bias.float()
        self.calls += 1
        return output.bfloat16().reshape(*x.shape[:-1], self.n)


class PreparedW8A8I8MMAddNorm:
    """Same-rounding W8 Linear + residual LayerNorm in one CPU TLE Raw team."""

    def __init__(self, linear, weight, bias, eps=1e-5, *, norm_threads=8):
        import math
        import struct

        if linear.kernel != "fused" or not 1 <= linear.n <= 4096:
            raise ValueError("Fused I8MM plan and norm width <=4096 required")
        for t in (weight, bias):
            if (
                t.device.type != "cpu"
                or t.dtype != torch.bfloat16
                or t.shape != (linear.n,)
                or not t.is_contiguous()
            ):
                raise ValueError("Contiguous CPU BF16 norm parameters required")
        if (
            not math.isfinite(eps)
            or eps <= 0
            or not isinstance(norm_threads, int)
            or not 1 <= norm_threads <= (os.cpu_count() or 1)
        ):
            raise ValueError("Positive epsilon and valid norm workers required")
        native = _entry("flag_gems_arm_w8a8_i8mm_fused")[0]
        abi = getattr(native, "flag_gems_arm_w8a8_i8mm_norm_abi_version", None)
        if abi is None or abi() != 1:
            raise RuntimeError("Build the optional I8MM residual norm ABI1 library")
        self.linear, self.weight, self.bias = linear, weight, bias
        self.norm_threads = norm_threads
        self.eps_bits = struct.unpack("I", struct.pack("f", eps))[0]
        self.entry = _entry("flag_gems_arm_w8a8_i8mm_fused_norm")
        self._tensor_raw_ops = linear._tensor_raw_ops

    def __call__(self, x, residual, bias=None, *, threads=None):
        plan = self.linear
        if self._tensor_raw_ops is not None:
            count = plan.threads if threads is None else threads
            output = self._tensor_raw_ops[1](
                x,
                residual,
                plan.packed,
                plan.scales,
                plan.inverse,
                bias,
                self.weight,
                self.bias,
                self.eps_bits,
                count,
                self.norm_threads,
                plan._order_for(x),
            )
            if not torch.compiler.is_compiling():
                plan.calls += 1
            return output
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or x.ndim < 2
            or x.shape[-1] != plan.k
        ):
            raise ValueError("CPU BF16 input ending in K required")
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
        if not isinstance(threads, int) or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Invalid thread count")
        flat = x.reshape(-1, plan.k).contiguous()
        m = flat.shape[0]
        output = torch.empty((m, plan.n), dtype=torch.bfloat16)
        if m:
            mp = (m + 7) // 8 * 8
            for name, count, dtype in [
                ("a", mp * plan.k, torch.int8),
                ("scale", mp, torch.float32),
            ]:
                value = getattr(_scratch, name, None)
                if value is None or value.numel() < count:
                    setattr(_scratch, name, torch.empty(count, dtype=dtype))
            descriptor = (ctypes.c_int64 * 18)(
                flat.data_ptr(),
                plan.packed.data_ptr(),
                plan.scales.data_ptr(),
                _scratch.a.data_ptr(),
                _scratch.scale.data_ptr(),
                output.data_ptr(),
                0 if bias is None else bias.data_ptr(),
                m,
                plan.n,
                plan.k,
                threads,
                0 if plan.inverse is None else plan.inverse.data_ptr(),
                int(plan._order_for(flat)),
                residual.data_ptr(),
                self.weight.data_ptr(),
                self.bias.data_ptr(),
                self.eps_bits,
                self.norm_threads,
            )
            self.entry[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
            plan.calls += 1
        return output.reshape(shape)


@lru_cache(maxsize=1)
def _ffn_tensor_raw_operator(library_path):
    from flag_gems.runtime.backend._arm.tle_raw.native_raw import initialize_native_raw

    initialize_native_raw(library_path)

    @torch.library.register_fake("flag_gems_arm_w8a8_i8mm::ffn.raw")
    def meta(
        x,
        uw,
        us,
        ui,
        ub,
        dw,
        ds,
        di,
        db,
        lut,
        nw,
        nb,
        eps,
        threads,
        norm_threads,
        up_n,
        down_n,
    ):
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    return torch.ops.flag_gems_arm_w8a8_i8mm.ffn.raw


class PreparedW8A8I8MMFFN:
    """Two prepared W8 linears, BF16 activation lookup and residual norm via TLE Raw."""

    def __init__(self, up, down_norm, activation_lut):
        if (
            up.kernel != "fused"
            or up._tensor_raw_ops is None
            or down_norm._tensor_raw_ops is None
            or up.n != down_norm.linear.k
            or up.k != down_norm.linear.n
            or up.threads != down_norm.linear.threads
        ):
            raise ValueError("Matching typed fused I8MM FFN plans required")
        if (
            activation_lut.device.type != "cpu"
            or activation_lut.dtype != torch.bfloat16
            or activation_lut.shape != (65536,)
            or not activation_lut.is_contiguous()
        ):
            raise ValueError("Contiguous CPU BF16 activation table [65536] required")
        native = _entry("flag_gems_arm_w8a8_i8mm_fused")[0]
        abi = getattr(native, "flag_gems_arm_w8a8_i8mm_ffn_abi_version", None)
        if abi is None or abi() != 1:
            raise RuntimeError("Build the typed I8MM FFN TLE Raw ABI1 library")
        self.up, self.down_norm, self.activation_lut = up, down_norm, activation_lut
        self.operator = _ffn_tensor_raw_operator(
            _entry("flag_gems_arm_w8a8_i8mm_fused")[0]._name
        )

    def __call__(self, x, up_bias=None, down_bias=None):
        u, norm = self.up, self.down_norm
        d = norm.linear
        out = self.operator(
            x,
            u.packed,
            u.scales,
            u.inverse,
            up_bias,
            d.packed,
            d.scales,
            d.inverse,
            down_bias,
            self.activation_lut,
            norm.weight,
            norm.bias,
            norm.eps_bits,
            u.threads,
            norm.norm_threads,
            u._order_for(x),
            d._order_for(x),
        )
        if not torch.compiler.is_compiling():
            u.calls += 1
            d.calls += 1
        return out
