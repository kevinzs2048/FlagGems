"""Native Arm image operators, owned by FlagGems; no model or device registration."""

import hashlib
import os
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock
from torch.utils.cpp_extension import load

SOURCE_DIR = Path(__file__).resolve().parents[4] / "csrc/arm/qwen_image"


def kleidiai_root():
    value = os.environ.get("FLAGGEMS_KLEIDIAI_ROOT")
    if not value:
        raise RuntimeError(
            "Set FLAGGEMS_KLEIDIAI_ROOT to the KleidiAI v1.29.0 source/build directory"
        )
    root = Path(value).expanduser().resolve()
    if not (root / "build-perf/libkleidiai.a").is_file():
        raise RuntimeError(
            "Missing KleidiAI library: " + str(root / "build-perf/libkleidiai.a")
        )
    return root


def load_ops(name):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This native image profile currently requires macOS arm64")
    if name not in _LOADERS:
        raise ValueError("Unknown image operator: " + str(name))
    return _LOADERS[name]()


@lru_cache(None)
def _load_accelerate_threads():
    source = SOURCE_DIR / "accelerate_threads.cpp"
    load(
        name="qwen21_accelerate_threads",
        sources=[str(source)],
        extra_cflags=["-O3", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[
            "-framework",
            "Accelerate",
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_accelerate_threads


@lru_cache(None)
def _load_bf16_register_epilogue():
    _load_bf16_sme()
    root = kleidiai_root()
    project = SOURCE_DIR
    cache = (
        Path(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR", Path.home() / ".cache/torch_extensions"
            )
        )
        / "qwen_image_objects"
    )
    cache.mkdir(parents=True, exist_ok=True)
    assembly = project / "bf16_register_epilogue_asm.S"
    digest = hashlib.sha256(assembly.read_bytes()).hexdigest()[:16]
    obj = cache / f"bf16_register_epilogue_{digest}.o"
    with FileLock(str(obj) + ".lock"):
        if not obj.exists():
            temporary = obj.with_suffix(".tmp.o")
            subprocess.run(
                [
                    "xcrun",
                    "clang",
                    "-c",
                    "-march=armv8.6-a+sve2",
                    str(assembly),
                    "-o",
                    str(temporary),
                ],
                check=True,
            )
            temporary.replace(obj)
    load(
        name="qwen21_bf16_register_epilogue",
        sources=[str(project / "bf16_register_epilogue.cpp")],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-DNDEBUG", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[
            str(obj),
            str(root / "build-perf/libkleidiai.a"),
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_bf16_register_epilogue


@lru_cache(maxsize=1)
def _load_bf16_sme():
    for flag in ("hw.optional.arm.FEAT_SME2", "hw.optional.arm.SME_B16F32"):
        if subprocess.check_output(["sysctl", "-n", flag], text=True).strip() != "1":
            raise RuntimeError("SME2 BF16 matrix instructions are required")
    root = kleidiai_root()
    library = root / "build-perf/libkleidiai.a"
    source = SOURCE_DIR / "bf16_sme.cpp"
    load(
        name="qwen21_bf16_sme",
        sources=[str(source)],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-DNDEBUG", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[
            str(library),
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_bf16_sme


@lru_cache(None)
def _load_bf16_tile_epilogue():
    _load_bf16_sme()
    root = kleidiai_root()
    source = SOURCE_DIR / "bf16_tile_epilogue.cpp"
    load(
        name="qwen21_bf16_epilogue",
        sources=[str(source)],
        extra_include_paths=[str(root)],
        extra_cflags=[
            "-O3",
            "-ffp-contract=off",
            "-DNDEBUG",
            "-Xpreprocessor",
            "-fopenmp",
        ],
        extra_ldflags=[
            str(root / "build-perf/libkleidiai.a"),
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_bf16_epilogue


@lru_cache(None)
def _load_bf16_variants():
    _load_bf16_sme()  # Checks CPU capability and loads the original packer.
    root = kleidiai_root()
    source = SOURCE_DIR / "bf16_sme_variants.cpp"
    load(
        name="qwen21_bf16_variants",
        sources=[str(source)],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-DNDEBUG", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[
            str(root / "build-perf/libkleidiai.a"),
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_bf16_variants


@lru_cache(None)
def _load_cluster_tiles():
    _load_bf16_sme()
    root = kleidiai_root()
    source = SOURCE_DIR / "bf16_cluster_tiles.cpp"
    load(
        name="qwen21_bf16_cluster",
        sources=[str(source)],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-DNDEBUG", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[
            str(root / "build-perf/libkleidiai.a"),
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_bf16_cluster


@lru_cache(None)
def _load_layernorm_scale():
    source = SOURCE_DIR / "layernorm_scale.cpp"
    load(
        name="qwen21_layernorm_scale",
        sources=[str(source)],
        extra_cflags=[
            "-O3",
            "-DAT_BUILD_ARM_VEC256_WITH_SLEEF",
            "-Xpreprocessor",
            "-fopenmp",
        ],
        extra_ldflags=[f'-L{Path(torch.__file__).parent / "lib"}', "-lomp"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_layernorm_scale


@lru_cache(None)
def _load_norm_rope():
    source = SOURCE_DIR / "norm_rope.cpp"
    load(
        name="qwen21_norm_rope",
        sources=[str(source)],
        extra_cflags=["-O3", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[f'-L{Path(torch.__file__).parent / "lib"}', "-lomp"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_norm_rope


@lru_cache(None)
def _load_pointwise_fusion():
    source = SOURCE_DIR / "pointwise_fusion.cpp"
    load(
        name="qwen21_pointwise",
        sources=[str(source)],
        extra_cflags=["-O3", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[f'-L{Path(torch.__file__).parent / "lib"}', "-lomp"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_pointwise


@lru_cache(None)
def _load_qkv_mixed():
    _load_bf16_sme()
    if (
        subprocess.check_output(
            ["sysctl", "-n", "hw.optional.arm.FEAT_I8MM"], text=True
        ).strip()
        != "1"
    ):
        raise RuntimeError("Joint QKV requires NEON I8MM")
    root = kleidiai_root()
    source = SOURCE_DIR / "qkv_mixed.cpp"
    load(
        name="qwen21_qkv_mixed",
        sources=[str(source)],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-DNDEBUG", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[
            str(root / "build-perf/libkleidiai.a"),
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_qkv_mixed


@lru_cache(None)
def _load_rmsnorm_fusion():
    source = SOURCE_DIR / "rmsnorm_fusion.cpp"
    load(
        name="qwen21_rmsnorm",
        sources=[str(source)],
        extra_cflags=["-O3", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[f'-L{Path(torch.__file__).parent / "lib"}', "-lomp"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_rmsnorm


@lru_cache(None)
def _load_rope_fusion():
    source = SOURCE_DIR / "rope_fusion.cpp"
    load(
        name="qwen21_rope",
        sources=[str(source)],
        extra_cflags=["-O3", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[f'-L{Path(torch.__file__).parent / "lib"}', "-lomp"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_rope


@lru_cache(None)
def _load_w8_swiglu():
    _load_w8a8_sme()
    root = kleidiai_root()
    source = SOURCE_DIR / "w8_swiglu.cpp"
    load(
        name="qwen21_w8_swiglu",
        sources=[str(source)],
        extra_include_paths=[str(root)],
        extra_cflags=[
            "-O3",
            "-ffp-contract=off",
            "-DNDEBUG",
            "-Xpreprocessor",
            "-fopenmp",
        ],
        extra_ldflags=[
            str(root / "build-perf/libkleidiai.a"),
            f'-L{Path(torch.__file__).parent / "lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_w8_swiglu


@lru_cache(maxsize=1)
def _load_w8a8_sme():
    for flag in ("hw.optional.arm.FEAT_SME2", "hw.optional.arm.SME_I8I32"):
        if subprocess.check_output(["sysctl", "-n", flag], text=True).strip() != "1":
            raise RuntimeError("SME2 INT8 matrix instructions required")
    root = kleidiai_root()
    source = SOURCE_DIR / "w8a8_sme.cpp"
    load(
        name="qwen21_w8a8_sme",
        sources=[str(source)],
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-DNDEBUG", "-Xpreprocessor", "-fopenmp"],
        extra_ldflags=[
            str(root / "build-perf/libkleidiai.a"),
            f'-L{Path(torch.__file__).parent/"lib"}',
            "-lomp",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.qwen21_w8a8_sme


_LOADERS = {
    "accelerate_threads": _load_accelerate_threads,
    "bf16_register_epilogue": _load_bf16_register_epilogue,
    "bf16_sme": _load_bf16_sme,
    "bf16_tile_epilogue": _load_bf16_tile_epilogue,
    "bf16_variants": _load_bf16_variants,
    "cluster_tiles": _load_cluster_tiles,
    "layernorm_scale": _load_layernorm_scale,
    "norm_rope": _load_norm_rope,
    "pointwise_fusion": _load_pointwise_fusion,
    "qkv_mixed": _load_qkv_mixed,
    "rmsnorm_fusion": _load_rmsnorm_fusion,
    "rope_fusion": _load_rope_fusion,
    "w8_swiglu": _load_w8_swiglu,
    "w8a8_sme": _load_w8a8_sme,
}
