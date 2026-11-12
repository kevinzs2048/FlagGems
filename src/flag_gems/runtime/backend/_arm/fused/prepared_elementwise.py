# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared BF16 CPU operators with cached Triton and TLE Raw entries."""

import ctypes
import math
import os
import struct
import sys
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cpu.tle_ops import raw_call


@triton.jit(do_not_specialize_on_alignment=["descriptor"])
def _elementwise_raw(descriptor, SYMBOL: tl.constexpr):
    raw_call(SYMBOL, descriptor.to(tl.uint64), result_dtype=None)


@lru_cache(maxsize=4)
def _entry(symbol, library_path=None):
    suffix = "dylib" if sys.platform == "darwin" else "so"
    default = (
        Path(__file__).resolve().parents[1]
        / "tle_raw/lib"
        / f"libflag_gems_arm_elementwise.{suffix}"
    )
    path = Path(
        library_path or os.environ.get("FLAGGEMS_ARM_ELEMENTWISE_LIBRARY", str(default))
    ).resolve()
    if not path.is_file():
        raise RuntimeError(
            "Build FLAGGEMS_BUILD_ARM_ELEMENTWISE=ON before selecting these operators"
        )
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    getattr(native, symbol)
    compiled = _elementwise_raw.warmup(
        torch.empty(1, dtype=torch.int64), SYMBOL=symbol, grid=(1,), num_cpu_threads=1
    )
    library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    fn = getattr(library, compiled.name)
    fn.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6
    fn.restype = None
    return native, compiled, library, fn


def _check_input(x):
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Contiguous CPU BF16 input required")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("Inference only")


class PreparedAddLayerNormRaw:
    def __init__(self, weight, bias, eps=1e-5, *, threads=4):
        _check_input(weight.detach())
        _check_input(bias.detach())
        if (
            weight.ndim != 1
            or bias.shape != weight.shape
            or not 1 <= weight.numel() <= 4096
            or not math.isfinite(eps)
            or eps <= 0
            or not isinstance(threads, int)
            or threads < 1
        ):
            raise ValueError(
                "Expected BF16 norm parameters and positive epsilon/threads"
            )
        self.weight, self.bias = weight, bias
        self.n, self.threads = weight.numel(), threads
        self.eps_bits = struct.unpack("I", struct.pack("f", eps))[0]
        self.entry = _entry("flag_gems_arm_add_layer_norm_bf16")

    def __call__(self, x, residual):
        _check_input(x)
        _check_input(residual)
        if x.ndim < 2 or x.shape[-1] != self.n or residual.shape != x.shape:
            raise ValueError(
                "Matching BF16 residual shapes ending in norm width required"
            )
        if torch.is_grad_enabled() and (
            self.weight.requires_grad or self.bias.requires_grad
        ):
            raise RuntimeError("Inference only")
        y = torch.empty_like(x)
        if x.numel():
            descriptor = (ctypes.c_int64 * 9)(
                x.data_ptr(),
                residual.data_ptr(),
                self.weight.data_ptr(),
                self.bias.data_ptr(),
                y.data_ptr(),
                x.numel() // self.n,
                self.n,
                self.threads,
                self.eps_bits,
            )
            self.entry[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
        return y


class PreparedGeluErfRaw:
    def __init__(self, *, threads=8, reference_threads=None):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("Positive thread count required")
        self.threads = threads
        self.reference_threads = reference_threads
        if reference_threads is not None and (
            not isinstance(reference_threads, int) or reference_threads < 1
        ):
            raise ValueError("Positive reference thread count required")
        self.entry = _entry("flag_gems_arm_gelu_lookup_bf16")

    def __call__(self, x):
        _check_input(x)
        y = torch.empty_like(x)
        if x.numel():
            descriptor = (ctypes.c_int64 * 5)(
                x.data_ptr(),
                y.data_ptr(),
                x.numel(),
                self.threads,
                self.reference_threads or torch.get_num_threads(),
            )
            self.entry[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
        return y


@triton.jit(
    do_not_specialize=["N", "STRIPE"],
    do_not_specialize_on_alignment=["x", "table", "output", "N", "STRIPE"],
)
def _bf16_lut_kernel(
    x, table, output, N, STRIPE, BLOCK: tl.constexpr, ALIGNED: tl.constexpr
):
    begin = tl.program_id(0) * STRIPE
    input_ptr = x.to(tl.pointer_type(tl.uint16)) + begin
    output_ptr = output.to(tl.pointer_type(tl.uint16)) + begin
    base = tl.arange(0, BLOCK)
    for start in tl.range(0, STRIPE, BLOCK, loop_unroll_factor=16):
        offsets = base + start
        if ALIGNED:
            indices = tl.load(input_ptr + offsets).to(tl.int32)
            values = tl.load(table.to(tl.pointer_type(tl.uint16)) + indices)
            tl.store(output_ptr + offsets, values)
        else:
            indices = tl.load(input_ptr + offsets, begin + offsets < N, other=0).to(
                tl.int32
            )
            values = tl.load(table.to(tl.pointer_type(tl.uint16)) + indices)
            tl.store(output_ptr + offsets, values, begin + offsets < N)


@lru_cache(maxsize=16)
def _bf16_lut_entry(threads, aligned):
    grid = (threads, 1, 1)
    dummy = torch.empty(1, dtype=torch.bfloat16, device="cpu")
    compiled = _bf16_lut_kernel.warmup(
        dummy,
        dummy,
        dummy,
        # Warmup compiles without execution; force 64-bit runtime lengths.
        N=2**40,
        STRIPE=2**40,
        BLOCK=1,
        ALIGNED=aligned,
        grid=grid,
        num_cpu_threads=threads,
    )
    return compiled, compiled[grid]


class PreparedBF16Lookup:
    """Apply an immutable 65536-entry BF16 bit-pattern table with CPU Triton.

    No floating-point approximation is introduced: inputs select table entries
    by their 16-bit encoding, including NaNs and signed zeros. The caller owns
    the table's meaning and must keep it unchanged while this plan is used.
    Compiled launchers are cached; inference allocates a fresh result and
    executes one contiguous stripe per worker through the CPU grid launcher.
    """

    def __init__(self, table, *, threads=8):
        if (
            table.device.type != "cpu"
            or table.dtype != torch.bfloat16
            or table.shape != (65536,)
            or not table.is_contiguous()
        ):
            raise ValueError("Contiguous CPU BF16 table with 65536 entries required")
        if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Invalid lookup thread count")
        if table.requires_grad:
            raise ValueError("Lookup table is inference-only")
        self.table = table
        self.threads = threads

    def __call__(self, x):
        _check_input(x)
        output = torch.empty_like(x)
        if x.numel():
            # Keep all DSO owners alive through the synchronous call even if
            # another caller evicts this specialization from the bounded cache.
            # Two variants per thread count, independent of sequence length.
            # Do not accumulate a compiled kernel for every document length.
            stripe = triton.cdiv(x.numel(), self.threads)
            aligned = x.numel() == stripe * self.threads
            owners = _bf16_lut_entry(self.threads, aligned)
            owners[1](x, self.table, output, x.numel(), stripe, 1, aligned)
        return output


class PreparedBF16LookupRaw(PreparedBF16Lookup):
    """The same exact BF16 lookup through the CPU TLE Raw native leaf."""

    def __init__(self, table, *, threads=8, library_path=None):
        super().__init__(table, threads=threads)
        self.entry = _entry("flag_gems_arm_bf16_lookup", library_path)

    def __call__(self, x):
        _check_input(x)
        output = torch.empty_like(x)
        if x.numel():
            descriptor = (ctypes.c_int64 * 5)(
                x.data_ptr(),
                self.table.data_ptr(),
                output.data_ptr(),
                x.numel(),
                self.threads,
            )
            self.entry[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
        return output


@triton.jit(
    do_not_specialize=["N", "CHUNK"],
    do_not_specialize_on_alignment=[
        "x",
        "vector_table",
        "scalar_table",
        "output",
        "N",
        "CHUNK",
    ],
)
def _gelu_bf16_tables_kernel(
    x,
    vector_table,
    scalar_table,
    output,
    N,
    CHUNK,
    VECTOR_SPAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    begin = tl.program_id(0) * CHUNK
    end = tl.minimum(begin + CHUNK, N)
    vector_count = tl.maximum(end - begin, 0) // VECTOR_SPAN * VECTOR_SPAN
    for start in tl.range(0, vector_count, BLOCK, loop_unroll_factor=8):
        offsets = begin + start + tl.arange(0, BLOCK)
        indices = tl.load(x.to(tl.pointer_type(tl.uint16)) + offsets).to(tl.int32)
        values = tl.load(vector_table.to(tl.pointer_type(tl.uint16)) + indices)
        tl.store(output.to(tl.pointer_type(tl.uint16)) + offsets, values)
    offsets = begin + vector_count + tl.arange(0, VECTOR_SPAN)
    indices = tl.load(
        x.to(tl.pointer_type(tl.uint16)) + offsets, offsets < end, other=0
    ).to(tl.int32)
    values = tl.load(scalar_table.to(tl.pointer_type(tl.uint16)) + indices)
    tl.store(output.to(tl.pointer_type(tl.uint16)) + offsets, values, offsets < end)


@lru_cache(maxsize=16)
def _gelu_bf16_tables_entry(threads, tasks, vector_span):
    dummy = torch.empty(1, dtype=torch.bfloat16, device="cpu")
    grid = (tasks, 1, 1)
    compiled = _gelu_bf16_tables_kernel.warmup(
        dummy,
        dummy,
        dummy,
        dummy,
        N=2**40,
        CHUNK=2**40,
        VECTOR_SPAN=vector_span,
        # Scalar lookup addresses avoid vector-to-scalar address shuffles on
        # Arm cores without a native general NEON gather instruction.
        BLOCK=1,
        grid=grid,
        num_cpu_threads=threads,
    )
    return compiled, compiled[grid]


class PreparedGeluErfTriton:
    """BF16 erf-GELU in Triton with ATen's exact vector/scalar partition.

    A one-time TLE Raw preparation captures the two complete BF16 mappings.
    Every forward executes the Triton lookup kernel, including scalar tails.
    No per-length kernel or per-layer table is required: share this plan across
    compatible GELU layers. The two immutable tables occupy 256 KiB in total.
    """

    def __init__(self, *, threads=8, reference_threads=None, library_path=None):
        if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Invalid GELU thread count")
        if reference_threads is not None and (
            type(reference_threads) is not int or reference_threads < 1
        ):
            raise ValueError("Positive reference thread count required")
        self.threads = threads
        self.reference_threads = reference_threads
        self.vector_table = torch.empty(65536, dtype=torch.bfloat16, device="cpu")
        self.scalar_table = torch.empty_like(self.vector_table)
        vector_span = ctypes.c_int64()
        descriptor = (ctypes.c_int64 * 3)(
            self.vector_table.data_ptr(),
            self.scalar_table.data_ptr(),
            ctypes.addressof(vector_span),
        )
        owners = _entry("flag_gems_arm_prepare_gelu_bf16_tables", library_path)
        owners[3](ctypes.addressof(descriptor), 0, 0, 0, 1, 1, 1)
        self.vector_span = vector_span.value
        if self.vector_span < 2 or self.vector_span & (self.vector_span - 1):
            raise RuntimeError("Unsupported ATen BF16 vector width")

    def __call__(self, x):
        _check_input(x)
        output = torch.empty_like(x)
        if x.numel():
            reference = self.reference_threads or torch.get_num_threads()
            grain = x.numel() // reference if x.numel() > 16384 else 32768
            if grain < 1:
                raise ValueError("Reference thread count exceeds the element count")
            tasks = (
                min(reference, triton.cdiv(x.numel(), grain))
                if x.numel() > grain
                else 1
            )
            chunk = triton.cdiv(x.numel(), tasks)
            owners = _gelu_bf16_tables_entry(self.threads, tasks, self.vector_span)
            owners[1](
                x,
                self.vector_table,
                self.scalar_table,
                output,
                x.numel(),
                chunk,
                self.vector_span,
                1,
            )
        return output
