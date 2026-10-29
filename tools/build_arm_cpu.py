# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Build optional Arm CPU operators from a FlagGems source checkout."""

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("silu", "standalone", "kai"), default="standalone"
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--install-prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--kai-root", type=Path)
    parser.add_argument("--kai-library", type=Path)
    parser.add_argument("--bitcode-compiler", type=Path)
    parser.add_argument("--openmp-include", type=Path)
    args = parser.parse_args()
    if platform.machine().lower() not in ("arm64", "aarch64"):
        parser.error("Arm CPU operators require an AArch64 host")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.mode == "kai" and not (args.kai_root and args.kai_library):
        parser.error("--mode kai requires --kai-root and --kai-library")
    import torch

    source = (
        Path(__file__).resolve().parents[1]
        / "src/flag_gems/runtime/backend/_arm/tle_raw/csrc"
    )
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(args.build_dir.resolve()),
        f"-DFLAGGEMS_BUILD_ARM_KAI={'ON' if args.mode == 'kai' else 'OFF'}",
        f"-DFLAGGEMS_BUILD_ARM_STANDALONE={'ON' if args.mode == 'standalone' else 'OFF'}",
        "-DFLAGGEMS_INSTALL=ON",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DTorch_DIR={Path(torch.__file__).parent / 'share/cmake/Torch'}",
    ]
    if args.kai_root:
        command.append(f"-DKLEIDIAI_ROOT={args.kai_root.resolve()}")
    if args.kai_library:
        command.append(f"-DKLEIDIAI_LIBRARY={args.kai_library.resolve()}")
    if args.bitcode_compiler:
        command.append(
            f"-DFLAGGEMS_ARM_BITCODE_COMPILER={args.bitcode_compiler.resolve()}"
        )
    if args.openmp_include:
        if sys.platform != "darwin":
            parser.error("--openmp-include is for Apple Clang")
        omp = Path(torch.__file__).parent / "lib/libomp.dylib"
        command += [
            f"-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I{args.openmp_include.resolve()}",
            "-DOpenMP_CXX_LIB_NAMES=omp",
            f"-DOpenMP_omp_LIBRARY={omp}",
        ]
    subprocess.run(command, check=True)
    subprocess.run(
        ["cmake", "--build", str(args.build_dir.resolve()), "-j", str(args.jobs)],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--install",
            str(args.build_dir.resolve()),
            "--prefix",
            str(args.install_prefix.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
