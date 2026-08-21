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

## Coarse prefill scheduling

The W4 G128 and W8 prefill kernels have an optional coarse-N scheduling mode.
Instead of exposing every M-by-N4 tile as an independent CPU program, one
worker claims a short contiguous N4 stripe and runs all M16 tiles for that
stripe inside the Triton program.  The C++ operator only allocates the atomic
counter and launches the grid; packing, I8MM accumulation and epilogue math
remain in the versioned Triton sources.

The generic library default is disabled because the profitable stripe size is
CPU- and shape-dependent.  A platform profile may enable a measured policy:

- W4 G128: `FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL=1` and
  `FLAGGEMS_ARM_Q4_G128_STEAL_CHUNK={1,2,4,8,16,32}`.
- W8: `FLAGGEMS_W8_STEALING_PREFILL=1` and
  `FLAGGEMS_W8_PREFILL_STEAL_CHUNK=<1..32>`.

Decode is selected independently and continues to use the SDOT kernels for
single-token inputs.  The Arm runtime tests compare regular-grid and
coarse-stripe outputs bit-for-bit before a deployment profile enables either
prefill policy.
