# FlagGems Arm native operators

This directory owns the small C++ dispatcher used by the Arm CPU Q4, W8 and
GDN paths. Matrix kernels remain ordinary Triton programs imported from the
FlagGems Arm backend; the shared library registers their stable PyTorch CPU
operator ABI and owns the recurrent GDN loops that do not yet win as Triton
kernels.

The build requires a CPU-enabled `libtriton_jit` package. A standalone local
build is useful while developing the operator bundle:

```bash
cmake -S src/flag_gems/csrc/arm -B build/arm-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_JIT_ROOT=/path/to/libtriton_jit \
  -DTRITON_JIT_BUILD=/path/to/libtriton_jit/build
cmake --build build/arm-native --parallel
```

For a wheel build, enable FlagGems C extensions with backend `ARM`, select the
external Triton JIT package, and pass its `TritonJIT_DIR`. The library and the
two Python kernel-entry modules are installed together under
`flag_gems/csrc/arm`; `configure_runtime()` resolves them without developer
absolute paths.
