# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared W8A8 SME2 matrix operator through a cached CPU Triton Raw entry."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import threading
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call


@triton.jit(do_not_specialize_on_alignment=["descriptor"])
def _w8a8_sme(descriptor, SYMBOL: tl.constexpr):
    raw_call(SYMBOL, descriptor.to(tl.uint64), result_dtype=None)


@lru_cache(maxsize=32)
def _entry(scaled=False, profile=False, fused=False, hybrid=False, panel=False):
    from ..fused.bf16_sme import _entry as require_sme

    require_sme()
    if panel and not hybrid:
        raise ValueError("Panel packing requires the hybrid W8 kernel")
    library = "libflag_gems_arm_w8a8_sme" + ("_panel" if panel else "") + ".dylib"
    variable = (
        "FLAGGEMS_ARM_W8_PANEL_LIBRARY" if panel else "FLAGGEMS_ARM_W8_SME_LIBRARY"
    )
    path = Path(
        os.environ.get(
            variable, str(Path(__file__).resolve().parents[1] / "lib" / library)
        )
    )
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    if panel and (
        not hasattr(native, "flag_gems_arm_w8a8_panel_abi_version")
        or native.flag_gems_arm_w8a8_panel_abi_version() != 1
    ):
        raise RuntimeError("Panel packing requires the independent ABI1 W8 leaf")
    symbol = (
        "flag_gems_arm_w8a8_sme"
        + ("_panel" if panel else "")
        + ("_scaled" if scaled else "")
        + ("_hybrid" if hybrid else "_fused" if fused else "")
        + ("_profile" if profile else "")
    )
    if not hasattr(native, symbol):
        raise RuntimeError(f"Rebuild the FlagGems W8 SME library: missing {symbol}")
    compiled = _w8a8_sme.warmup(
        torch.empty(1, dtype=torch.uint8), SYMBOL=symbol, grid=(1,), num_cpu_threads=1
    )
    dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    fn = getattr(dso, compiled.name)
    fn.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    fn.restype = None
    return native, compiled, dso, fn


_scratch = threading.local()


@lru_cache(maxsize=1)
def _tensor_linear_entry():
    from ..tle_raw.native_raw import initialize_native_raw

    initialize_native_raw(os.environ["FLAGGEMS_ARM_FFN_TEAM_LIBRARY"])
    op = torch.ops.flag_gems_arm_w8a8_sme.linear.raw

    @torch.library.register_fake("flag_gems_arm_w8a8_sme::linear.raw")
    def fake(
        x, w, wi, scales, inverse, bias, residual, nw, nb, eps_bits, threads, sme, nt
    ):
        return torch.empty(
            (*x.shape[:-1], scales.numel()), dtype=x.dtype, device=x.device
        )

    return op


class PreparedW8A8SME:
    """Symmetric channel W8 and token A8, with caller-selected CPU scheduling.

    Fused variants write BF16 directly. Hybrid owns a second weight layout and
    shares disjoint output tiles between SME and Neon I8MM workers. The number
    of SME workers is a software scheduling choice, not a hardware unit count.
    Optional panel packing uses an independent CPU TLE leaf for33..64 rows;
    other shapes retain the ordinary entry and its packed formats.
    """

    def __init__(
        self,
        weight,
        threads=8,
        smoothing=None,
        *,
        kernel="scratch",
        sme_threads=3,
        panel_packing=False,
    ):
        if (
            weight.device.type != "cpu"
            or weight.ndim != 2
            or not weight.is_floating_point()
            or weight.shape[1] % 4
        ):
            raise ValueError("Expected a CPU floating weight [N,K], K divisible by4")
        if not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Invalid threads")
        if kernel not in {"scratch", "fused", "hybrid"}:
            raise ValueError("Invalid W8 SME kernel")
        self.kernel = kernel
        if type(panel_packing) is not bool or panel_packing and kernel != "hybrid":
            raise ValueError("Optional panel packing requires the hybrid W8 kernel")
        self.panel_packing = panel_packing
        self.sme_threads = sme_threads
        if kernel == "hybrid" and (
            weight.shape[1] % 8
            or not isinstance(sme_threads, int)
            or not 0 <= sme_threads <= threads
        ):
            raise ValueError(
                "Hybrid requires K divisible by8 and valid SME thread count"
            )
        _entry(
            smoothing is not None,
            fused=kernel.startswith("fused"),
            hybrid=kernel == "hybrid",
        )
        if panel_packing:
            _entry(smoothing is not None, hybrid=True, panel=True)
        self.n, self.k = weight.shape
        self.threads = threads
        self.calls = 0
        self.profile_enabled = False
        self.profile_rows = []
        values = weight.detach().float()
        self.inverse = None
        if smoothing is not None:
            if (
                smoothing.shape != (self.k,)
                or not torch.isfinite(smoothing).all()
                or (smoothing <= 0).any()
            ):
                raise ValueError("Invalid smoothing scale")
            values = values * smoothing.float()[None, :]
            self.inverse = smoothing.float().reciprocal().contiguous()
        if not torch.isfinite(values).all():
            raise ValueError("Finite weights required")
        self.scales = (values.abs().amax(1) / 127).clamp_min(1e-8).contiguous()
        q = (values / self.scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        np = (self.n + 63) // 64 * 64
        padded = F.pad(q, (0, 0, 0, np - self.n))
        self.packed_i8mm = (
            padded.reshape(np // 8, 8, self.k // 8, 8).permute(0, 2, 1, 3).contiguous()
            if kernel == "hybrid"
            else None
        )
        self.packed = (
            padded.reshape(np // 64, 64, self.k // 4, 4)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def __call__(self, x, bias=None, *, threads=None, sme_threads=None):
        if (
            os.environ.get("FLAGGEMS_ARM_SME_LINEAR_TENSOR_RAW") == "1"
            and self.kernel == "hybrid"
            and not self.profile_enabled
            and not self.panel_packing
        ):
            if getattr(self, "tensor_op", None) is None:
                self.tensor_op = _tensor_linear_entry()
            result = self.tensor_op(
                x,
                self.packed,
                self.packed_i8mm,
                self.scales,
                self.inverse,
                bias,
                None,
                None,
                None,
                0,
                self.threads if threads is None else threads,
                self.sme_threads if sme_threads is None else sme_threads,
                0,
            )
            if not torch.compiler.is_compiling() and x.numel():
                self.calls += 1
            return result
        started = time.perf_counter() if self.profile_enabled else 0
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or x.ndim < 1
            or x.shape[-1] != self.k
        ):
            raise ValueError("CPU BF16 input ending in K required")
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
        threads = self.threads if threads is None else threads
        sme_threads = self.sme_threads if sme_threads is None else sme_threads
        if not isinstance(threads, int) or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Invalid thread override")
        if self.kernel == "hybrid" and (
            not isinstance(sme_threads, int) or not 0 <= sme_threads <= threads
        ):
            raise ValueError("Invalid SME worker override")
        flat = x.reshape(-1, self.k).contiguous()
        m = flat.shape[0]
        mp = (m + 15) // 16 * 16
        np = (self.n + 63) // 64 * 64
        output = torch.empty((m, self.n), dtype=x.dtype)
        if m:
            for key, size, dtype in [
                ("a", mp * self.k, torch.int8),
                ("scale", mp, torch.float32),
                (
                    "c",
                    (
                        0
                        if self.kernel.startswith("fused") or self.kernel == "hybrid"
                        else mp * np
                    ),
                    torch.int32,
                ),
                ("ai", mp * self.k if self.kernel == "hybrid" else 0, torch.int8),
            ]:
                value = getattr(_scratch, key, None)
                if value is None or value.numel() < size:
                    setattr(_scratch, key, torch.empty(size, dtype=dtype))
            timing = (
                torch.empty(6 if self.kernel == "hybrid" else 4, dtype=torch.float64)
                if self.profile_enabled
                else None
            )
            descriptor = torch.tensor(
                [
                    flat.data_ptr(),
                    self.packed.data_ptr(),
                    self.scales.data_ptr(),
                    _scratch.a.data_ptr(),
                    _scratch.scale.data_ptr(),
                    _scratch.c.data_ptr(),
                    output.data_ptr(),
                    0 if bias is None else bias.data_ptr(),
                    m,
                    self.n,
                    self.k,
                    threads,
                    0 if self.inverse is None else self.inverse.data_ptr(),
                    0 if timing is None else timing.data_ptr(),
                    0 if self.packed_i8mm is None else self.packed_i8mm.data_ptr(),
                    _scratch.ai.data_ptr(),
                    sme_threads,
                ],
                dtype=torch.int64,
            )
            _entry(
                self.inverse is not None,
                self.profile_enabled,
                self.kernel.startswith("fused"),
                self.kernel == "hybrid",
                self.panel_packing and 32 < m <= 64,
            )[3](descriptor.data_ptr(), 0, 0, 0, 1, 1, 1)
            self.calls += 1
            if timing is not None:
                self.profile_rows.append(
                    {
                        "m": m,
                        "n": self.n,
                        "k": self.k,
                        "quantize_pack_s": timing[0].item(),
                        "sme_compute_s": timing[1].item(),
                        "epilogue_s": timing[2].item(),
                        "native_s": timing[3].item(),
                        "python_total_s": time.perf_counter() - started,
                    }
                )
            if timing is not None and self.kernel == "hybrid":
                self.profile_rows[-1].update(
                    sme_tiles=timing[4].item(), i8mm_tiles=timing[5].item()
                )
        return output.reshape(*x.shape[:-1], self.n)
