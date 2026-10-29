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

Set -DFLAGGEMS_BUILD_ARM_STANDALONE=ON for independent W8A8; K%32=0 and N%4=0.
Use arm_native_ops("w8a8", raw=True). Dynamic asymmetric per-token A8 and FP32
channel scales use the package-owned N4 format. Do not mix packed ABIs.
Torch supplies the matched OpenMP runtime; Apple Clang may require explicit
OpenMP compile/include flags and its runtime path. See the CMake cache options.
