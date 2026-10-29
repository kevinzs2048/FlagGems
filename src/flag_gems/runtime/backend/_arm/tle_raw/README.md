# Arm CPU TLE Raw operators

This optional AArch64 package owns its compiled Triton entries, native C ABI
leaves, FakeTensor schemas and independently built libraries. Native overloads
are explicit comparison routes. Raw still requires native source compilation.

Build with the Python/Torch environment used for inference:

```sh
cmake -S src/flag_gems/runtime/backend/_arm/tle_raw/csrc -B build/arm \
  -DPython_EXECUTABLE=/path/to/python
cmake --build build/arm -j4
cmake --install build/arm --prefix /path/to/site-packages
```

SiLU/up/A8 supports CPU FP32 rows, K divisible by 32, MR=1/4, and positive row
strides with inner stride one. The cached Raw entry uses FlagTree-CPU scalar
C ABI calls and the checked single-program scalar-return interface. Build
optional bitcode with FLAGGEMS_ARM_BITCODE_COMPILER from that LLVM toolchain.
Run tests/test_arm_native.py with the matched libraries installed or supplied
through FLAGGEMS_ARM_SILU_LIBRARY. No external integer microkernel is needed.
