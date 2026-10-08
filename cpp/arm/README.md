# Optional Arm CPU operators

This target keeps Arm inference arithmetic in FlagGems. It provides W4A8 G32
and per-channel W8A8 linear operators, routed MoE, fused SiLU/up/A8 packing,
and gated delta network (GDN) recurrence/convolution/normalization operators.
Both Darwin AArch64 and Linux AArch64 are supported. The target is opt-in and
does not change the default dispatcher or existing GPU builds.

## Build

Use the same Python and Torch environment as inference. Set
`FLAGGEMS_VENDOR=arm` to select the CPU backend, including on Linux systems
without an accelerator device. The dense kernels
require KleidiAI 1.29 headers and its AArch64 static library; routed experts use
the packed-weight ABI and the exact KAI functions exported by that Torch build.
Missing required ISA features or exported symbols cause a Python error rather
than executing unsupported instructions. SME2 is optional; Neon-only callers
use `split=0` / `sme=False`.

```sh
cmake -S cpp -B build/arm-native \
  -DFLAGGEMS_BUILD_C_EXTENSIONS=OFF \
  -DFLAGGEMS_BUILD_ARM_NATIVE=ON \
  -DFLAGGEMS_BUILD_ARM_KAI=ON \
  -DPython_EXECUTABLE=/path/to/python \
  -DTorch_DIR=/path/to/site-packages/torch/share/cmake/Torch \
  -DKLEIDIAI_ROOT=/path/to/KleidiAI \
  -DKLEIDIAI_LIBRARY=/path/to/libkleidiai.a
cmake --build build/arm-native -j4
cmake --install build/arm-native --prefix /path/to/site-packages
```

For Apple Clang, supply the existing toolchain's OpenMP include/compile flags
through `OpenMP_CXX_FLAGS`, and its runtime through `OpenMP_omp_LIBRARY` when
CMake needs them to find OpenMP. Inference links Torch's own OpenMP runtime to
avoid creating a second thread pool. `FLAGGEMS_BUILD_ARM_NATIVE=ON` alone builds
the SiLU packing library without requiring KAI.

Development builds may set `FLAGGEMS_ARM_NATIVE_LIBRARY` and
`FLAGGEMS_ARM_SILU_LIBRARY` to the corresponding built shared libraries. These
are explicit library-location overrides, not model adapters or JIT sources.

## Native and CPU TLE entries

```python
from flag_gems.runtime.backend._arm.fused import arm_native_ops

native = arm_native_ops("w4a8_g32")
raw = arm_native_ops("w4a8_g32", raw=True)
linear = raw.linear  # initialize once and retain the callable
```

Groups are `w4a8_g32`, `w8a8`, `moe` and `gdn`. Preparation and thread-policy
helpers retain their native entry. Inference methods expose native and `.raw`
Torch overloads. CPU Raw requires FlagTree CPU's scalar `raw_call` support with
void returns (the `triton_v3.7.x` / 3.7.2 native-call port). SiLU also requires
the optional single-program scalar-return C entry described below. A CPU build lacking
that optional capability can still import FlagGems and use native operators.

The chain is a compiled `@triton.jit` entry, CPU `raw_call`, then C++/Neon/KAI.
It is a CPU native-call API, not source-region embedding. Warmup retains each
compiled DSO and registers its compute address once. C++ calls that address
synchronously; inference does not use a Python Triton launcher, per-expert
lookup, or per-call JIT lookup. Dense host wrappers validate and allocate tensors before entering either the
native or Raw computation. Both entries execute the same C ABI computation
plan containing pointers, shapes and KAI kernels; no Tensor return value or
ownership transfer crosses this boundary. Each call includes A8 packing and
all GEMM tiles. Composite MoE/GDN operations cross the boundary once per
complete operation.
The C ABI catches C++ exceptions and the Torch caller rethrows them after the
compiled entry returns. Tensor references and descriptors stay alive until
all parallel work completes. The compiled Raw entry symbols and native
library are an internal matched ABI: rebuild the library and restart the
process when updating this implementation.

Neon-only single-row G32 calls pack A8 before the OpenMP fork, avoiding a
packing worksharing loop and barrier with only one useful iteration. Equal
128-column tiles with at most 32 rows use contiguous per-worker ownership
based on the actual OpenMP team size. Mixed SME2/Neon and partial-tile work
retain dynamic scheduling. Other noncanonical G32 packing paths synchronize
once after writing their disjoint LHS regions.
Explicit SME/Neon partitioning uses the actual OpenMP team size. If the
runtime grants one worker, that worker computes both output partitions;
otherwise at least one worker remains assigned to each partition.

GDN recurrence preserves the historical compiler contraction policy (GCC
`fast`, Clang/AppleClang `on`) for its multiply/add expressions. Native and
TLE Raw share the same recurrence body and FP32 state updates. SiLU packing
and the other sources keep `-ffp-contract=off`; applying that setting to GDN
would remove its native FMA instructions and change state rounding.
The ordinary Triton width-four convolution retains the historical
`enable_fp_fusion=True` compilation policy. This is independent of the native
SiLU packing policy.
Its prepared programs use the historical static OpenMP distribution, with
the team bounded by the available programs and caller's thread setting.
Nested parallel calls execute serially. This keeps all workers useful when
the program count is not divisible by the thread count, including 128
programs on an 18-thread CPU.

Linear inputs are contiguous FP32 `[M,K]`; W4 G32 scales are BF16, W8 scales
are FP32. `linear_a8` consumes canonical rows rather than requantizing them.
Dense KAI activation packing uses its ties-away schedule. Torch-packed routed
experts preserve that runtime's quantization and rounding; fused SiLU packing
uses RNE and is tested against the unfused expert arithmetic. These byte
formats must not be interchanged merely because both use A8 activations.

SME/Neon splits and worker counts are explicit arguments supplied by the
caller. The optional `sme_workers=-1` Apple 18-CPU cluster policy requires
`FLAGGEMS_ARM_APPLE_18CPU_CLUSTER_POLICY=1`; Linux uses explicit worker counts.
`FLAGGEMS_ARM_MOE_DECODE_THREADS` selects expert parallelism (0 uses the Torch
parallel policy). Set thread policy before inference workers start.

## Ordinary Triton convolution

Width-four BF16 causal convolution prefill and decode reuse ordinary Triton
arithmetic:

```python
from flag_gems.runtime.backend._arm.gdn import (
    gdn_conv1d_prefill_triton, gdn_conv1d_update_triton,
)

out = gdn_conv1d_prefill_triton(
    input, conv_state, weight, bias, query_start_loc,
    cache_indices, has_initial_state, silu_activation=True,
)
# Decode input is [tokens, dim], with history [slots, dim, 3].
# Each token must select a distinct cache slot.
out = gdn_conv1d_update_triton(
    decode_input, decode_state, weight, bias, state_indices,
    silu_activation=True,
)
```

This entry preserves arbitrary positive tensor strides, updates the first
three causal-history positions, and leaves speculative rollback history and
unselected cache slots unchanged. New empty sequences clear causal history;
existing empty sequences retain it. Compilation is cached outside the compute
path. For prewarmed C++ callers, prepare the dimension/bias/activation variant
using `prepare_conv_prefill`, then retain the `.triton` Torch overload.
The decode wrapper treats each token as a one-token sequence with existing
history and caches its read-only sequence metadata. It supports positive
noncontiguous strides, partial channel blocks and empty batches.
Cached convolution entries do not specialize on the alignment of their dummy
warmup tensors. Offset data and metadata views retain their documented support.

The SiLU exponential uses the SLEEF-u10 polynomial with its accompanying Boost
license. BF16 convolution outputs can differ from scalar `expf` by one BF16
step; state updates are exact. Ordinary matrix/GDN decode experiments are not
included as active paths until their shape-specific performance and complete
mutation contracts satisfy the same acceptance criteria.

## Validation and performance gate

Run `pytest -s --tb=short tests/test_arm_native.py` in the inference environment
with the optional libraries built. It covers independent quantization/linear
references, partial tiles, shared A8, SME/Neon combinations where supported,
router ties and renormalization, MoE scheduling variants, strided SiLU packing,
empty/multiple convolution sequences, state ownership and exception recovery.

Use this process for each operator and platform:

1. Freeze the native reference, packed ABI, numerical schedule, state mutations,
   matrix shapes and thread counts.
2. Reuse or implement ordinary Triton arithmetic first. Validate independently
   before measuring. Keep a rejected experiment outside the active backend.
3. If ordinary Triton cannot meet the performance/contract requirement, use
   a compiled CPU TLE entry around the complete native operation. Do not
   add a launch, lookup or synchronization for each expert or GEMM tile.
4. Check independent numerical references, tails, empty inputs, routing,
   mutable state, and error recovery on Darwin and Linux AArch64.
5. Warm compilation and retain launch addresses. Randomly interleave paired
   timing blocks; retain samples and source/build identity. Measure both the
   public entry and cached C++ calls when launch overhead matters.
6. Accept a measured latency regression strictly below 0.5%. Investigate larger
   regressions using generated IR, disassembly, allocation/copy inspection and
   thread/runtime profiling. Preserve failed runs; do not select the best rerun.
7. Promote only validated configurations. Commit functional code, tests and
   build documentation; keep weights, quantization generators, probes and
   experimental prototypes outside the operator repository.

Performance acceptance compares equivalent warmed native and selected Triton
or Raw operations with identical shapes, inputs and thread counts. Interleave
calls randomly, retain paired samples and investigate regressions of 0.5% or
more. A noisy confidence interval crossing that threshold requires further
measurement. Microbenchmarks do not establish whole-model PP512/TG128 speed;
model integration must also verify its complete prefill/decode path.

### SiLU Raw call boundary

The SiLU warmup requests `enable_single_program_entry=True` and registers
`compiled.metadata.single_program_name`. This additional compiler-generated
C entry fixes program IDs to zero and the grid to `(1,1,1)`. Its ABI contains
the six input arguments. With `single_program_return_type="i32"`, the final
status output pointer becomes a C return value, also avoiding six unused grid
arguments on every cached C++ call. The normal Triton kernel still writes
its output status tensor; LLVM promotes the additional entry's local scalar
to a register return after inlining. The regular compiled entry and normal Triton launcher retain
their full grid behavior. Without linked bitcode, the additional entry tail-branches to a non-throwing C ABI
leaf that returns its status result. The direct native and external Raw entries inline the same arithmetic
and share one scratch row per thread within their native library. Linked
arithmetic variants own separate per-thread scratch storage, retained by their
compiled shared libraries. Boundary-only variants call the native scratch owner. Allocation failures are caught before
returning through the generated entry. The external SLEEF C math function is
declared non-throwing, so an enclosing exception handler does not force the
compiler to spill live vector values on every loop iteration. Error recovery,
strided rows, rounding and the large-input SLEEF fallback are tested.

For small single-row calls, an optional LLVM bitcode build lets the CPU JIT
inline the same native arithmetic into the Triton entry. The default
`-DFLAGGEMS_ARM_SILU_BITCODE_MODE=kernel` selects this mode. Alternatively,
`-DFLAGGEMS_ARM_SILU_BITCODE_MODE=boundary` links only the C++ exception
boundary into the Triton entry, retaining the arithmetic compiled in the
standalone native library. This avoids an extra call to an external exception
wrapper without recompiling the arithmetic with a different LLVM toolchain.
The boundary calls the uniquely named `flag_gems_arm_silu_standalone_leaf`;
it cannot resolve the composite Arm library's copy. ELF builds export an alias
at the native function's address; other object formats use a forwarding entry.
Choose a mode using measurements on the target platform, shape and compiler.
Neither mode silently replaces a failed compilation with native execution.

Build with
`-DFLAGGEMS_ARM_BITCODE_COMPILER=/path/to/matching/llvm/bin/clang++` and an
explicit `-DCMAKE_OSX_SYSROOT=/path/to/MacOSX.sdk` on Darwin. CMake uses `-fPIC`
and installs `flag_gems_arm_silu.bc` alongside the operator library. The default
bitcode CPU is `generic`; `-DFLAGGEMS_ARM_BITCODE_CPU=native` tunes to the build
machine and limits portability to compatible CPUs. The LLVM version, OS, ABI
and CPU features must match the FlagTree CPU inference environment.

Set `FLAGGEMS_ARM_SILU_BITCODE=/path/to/flag_gems_arm_silu.bc` before Raw
initialization to opt in. Warmup prepares single-row MR1 and MR4 entries with
the selected bitcode, plus a general entry using the C ABI leaf for multiple
rows. Kernel-mode entries own their arithmetic and per-thread scratch storage;
boundary-mode entries use the standalone native function's scratch storage.
The C++ operator selects by row count and packing layout. All three entries
preserve the same quantization and padded output bytes. For diagnostics,
`FLAGGEMS_ARM_SILU_SPECIALIZE_MR=0` links a single general entry instead.
Prepared entries do not specialize on dummy warmup pointer alignment; offset
Tensor views and stack descriptors remain valid inputs. Native execution does
not depend on bitcode. Without a bitcode path, Raw uses
the general external C leaf; this is an explicit configuration, not a runtime
fallback after a failed compilation. Registration checks native ABI version 1
and the compiler's scalar-return metadata. Concurrent callers cannot observe
partially registered entries. Updating a library or bitcode requires restarting
the process; prepared kernels retain their loaded code for their lifetime.

Padded MR4 rows are written directly as canonical A8 zero panels (code 127,
offset -127, scale 1), without evaluating SiLU or quantizing a scratch zero row.
Both native and Raw use direct zero-panel writes. The optional bitcode build
also coalesces single-row MR4 codes and padding into one output pass; the native
reference retains its previously validated packing loop. Tests compare the
complete bytes against independent padded-panel expectations.

The Python SiLU helper selects the named `pack_raw` operator. Its default
schema avoids the nonempty overload-name handling in Torch's Python binding;
`pack.raw` remains a compatible alias to the same C++ implementation. Both
use normal Torch operators, preserving Torch dispatch interception.

Only the standalone SiLU library exports this status entry. On Linux its
function references bind locally, preventing a separately loaded composite
Arm library from changing which implementation a call measures or executes.

Native execution remains the default. Select Raw only after validating the
actual caller, shapes and thread policy on the target system. Cached C++
measurements exclude Python overhead and cannot substitute for public-entry
or whole-model measurements. Keep benchmark samples and source/build hashes
with the review evidence rather than hard-coding a machine's performance or
CPU placement into the operator implementation.
