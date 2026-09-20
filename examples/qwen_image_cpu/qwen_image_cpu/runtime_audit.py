"""Read actual runtime state, not just requested environment variables."""

import ctypes
import os
from pathlib import Path

import torch


def snapshot():
    omp = ctypes.CDLL(str(Path(torch.__file__).parent / "lib/libomp.dylib"))
    names = [
        "omp_get_max_threads",
        "omp_get_dynamic",
        "omp_get_num_places",
        "omp_get_place_num",
        "omp_get_proc_bind",
        "kmp_get_blocktime",
    ]
    system = ctypes.CDLL(None)
    system._dyld_image_count.restype = ctypes.c_uint32
    system._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    system._dyld_get_image_name.restype = ctypes.c_char_p
    libraries = [
        system._dyld_get_image_name(i).decode()
        for i in range(system._dyld_image_count())
    ]
    libraries = [
        p
        for p in libraries
        if any(s in Path(p).name for s in ["libomp", "libgomp", "libiomp", "jemalloc"])
    ]
    system.malloc_zone_from_ptr.argtypes = [ctypes.c_void_p]
    system.malloc_zone_from_ptr.restype = ctypes.c_void_p
    system.malloc_get_zone_name.argtypes = [ctypes.c_void_p]
    system.malloc_get_zone_name.restype = ctypes.c_char_p
    probe = torch.empty(1024, 1024)
    zone = system.malloc_get_zone_name(system.malloc_zone_from_ptr(probe.data_ptr()))
    explicit_jemalloc = hasattr(
        torch.ops.qwen21_jemalloc, "owns"
    ) and torch.ops.qwen21_jemalloc.owns(probe)
    return {
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "openmp": {name: getattr(omp, name)() for name in names},
        "libraries": libraries,
        "tensor_malloc_zone": zone.decode() if zone else None,
        "tensor_explicit_jemalloc": explicit_jemalloc,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("OMP_", "KMP_", "GOMP_", "DYLD_", "MALLOC_", "Malloc"))
        },
    }


def configure(model, kernel_owned, workers):
    from .bf16_sme import ops as bf16_ops
    from .w8a8_sme import DynamicW8SMELinear
    from .w8a8_sme import ops as w8_ops

    layers = list(model.modules())
    if any(getattr(layer, "_bf16_sme_enabled", False) for layer in layers):
        bf16_ops().configure_parallel(kernel_owned, workers)
    if any(isinstance(layer, DynamicW8SMELinear) for layer in layers):
        w8_ops().configure_parallel(kernel_owned, workers)
    for layer in layers:
        if isinstance(layer, DynamicW8SMELinear):
            layer.kernel_owned = kernel_owned
            layer.workers = workers
        if getattr(layer, "_bf16_sme_enabled", False):
            layer._bf16_sme_kernel_owned = kernel_owned
            layer._bf16_sme_workers = workers
