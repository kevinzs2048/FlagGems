# Arm CPU TLE Raw operators

This package owns the Arm CPU Triton Raw entries and their compiled C ABI leaves. It provides W4A8 G32
and per-channel W8A8 linear operators, routed MoE, fused SiLU/up/A8 packing,
and gated delta network (GDN) recurrence/convolution/normalization operators.
The target implements Darwin AArch64 and Linux AArch64 paths. It is opt-in and
does not change the default dispatcher or existing GPU builds.

The `gdn` namespace also exposes `gemma_rmsnorm` and in-place
`gemma_fused_add_rmsnorm`, including `.raw` overloads. They compute
`x * inv_rms * (weight.float() + 1)` before the final BF16 cast. The fused
operator rounds the residual sum to BF16 before computing its variance,
and updates both input and residual in place. It retains the arithmetic of
the validated ARM Gemma path; this rounding order can differ from a generic
fused PyTorch norm. The host wrapper owns validation and output allocation;
one cached CPU Triton entry invokes the common C++ compute plan. The native
library and Raw helper must be rebuilt and deployed together.

## Ownership and execution

`native_raw.py` and `silu_a8_raw.py` contain the compiled Triton entry kernels.
`csrc/` contains their native arithmetic leaves and synchronous Torch host
bindings; `lib/` is the installation destination for matched leaf DSOs.
The optimized plugin defaults to CPU TLE Raw. Native overloads remain an
explicit reference/comparison route. They are not a second default backend.
A Raw call still needs its compiled native leaves; moving their sources here
does not remove compilation or the optional KAI dependency.

## Build

### Independent W8A8 / GDN build

For the MiniCPM-V-4.7-1B W8A8 CPU path, use
`FLAGGEMS_BUILD_ARM_STANDALONE=ON` and `FLAGGEMS_BUILD_ARM_KAI=OFF`.
No KleidiAI headers, library, download, or SDK mount is required. This target
builds the FlagGems-owned W8A8 packing/compute and GDN/Norm routines. It excludes
W4 G32 and routed MoE; it does not claim to remove their existing dependencies.

W8 uses dynamic asymmetric per-token INT8 with a zero point. Weight packing
stores four channels per block, INT32 channel sums and FP32 scales. Decode
uses Neon SDOT. Four-row prefill tiles also use SDOT by default. Independent
I8MM is an explicit experiment (`FLAGGEMS_STANDALONE_PREFILL_I8MM=1`), not an
assumed speedup. Each activation row is quantized once, contiguous N stripes
reuse the prepared weight layout, and a cached CPU TLE Raw entry invokes the
whole compute plan. No KAI symbols participate in this implementation.

The arithmetic remains compiled C++/Neon inside TLE Raw. This is not a claim
that Raw removed native source compilation. The ordinary CPU Triton reference
in `w8a8_standalone_triton.py` implements quantization and GEMV without an
external microkernel, for correctness and measured performance comparisons.
It is not selected as the tuned default.

```sh
cmake -S src/flag_gems/runtime/backend/_arm/tle_raw/csrc -B build/arm-standalone \
  -DFLAGGEMS_BUILD_ARM_KAI=OFF \
  -DFLAGGEMS_BUILD_ARM_STANDALONE=ON \
  -DFLAGGEMS_INSTALL=ON \
  -DPython_EXECUTABLE=/path/to/python
cmake --build build/arm-standalone -j4
cmake --install build/arm-standalone --prefix /path/to/site-packages
```

Use `arm_native_ops("w8a8", raw=True)` and the existing GDN Raw entry. The
selected library supplies the weight ABI. Restart the process when changing
the library and never reuse packed weights from the other build variant.

### KleidiAI W4/MoE build (separate opt-in)

Use the same Python and Torch environment as inference. Set
`FLAGGEMS_VENDOR=arm` to select the CPU backend, including on Linux systems
without an accelerator device. The dense kernels
require KleidiAI 1.29 headers and its AArch64 static library; routed experts use
the packed-weight ABI and the exact KAI functions exported by that Torch build.
Missing required ISA features or exported symbols cause a Python error rather
than executing unsupported instructions. SME2 is optional; Neon-only callers
use `split=0` / `sme=False`.

```sh
cmake -S src/flag_gems/runtime/backend/_arm/tle_raw/csrc -B build/arm-native \
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
avoid creating a second thread pool. `python tools/build_arm_cpu.py --mode silu` builds
the SiLU leaf library without requiring KAI.

Development builds may set `FLAGGEMS_ARM_NATIVE_LIBRARY` and
`FLAGGEMS_ARM_SILU_LIBRARY` to the corresponding built shared libraries. These
are explicit library-location overrides, not model adapters or JIT sources.

## Native and CPU TLE entries

```python
from flag_gems.runtime.backend._arm.fused import arm_native_ops

raw = arm_native_ops("w4a8_g32", raw=True)
linear = raw.linear  # initialize once and retain the callable
native_reference = arm_native_ops("w4a8_g32")  # explicit comparison only
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
packing worksharing loop and barrier with only one useful iteration. For small
Neon calls, the requested worker count is an upper bound: the team is limited
to the larger of the useful packing tasks and GEMM tiles, avoiding idle workers
that compete for CPUs and wait at barriers. Equal
128-column tiles with at most 32 rows use contiguous per-worker ownership
based on the actual OpenMP team size. Mixed SME2/Neon and partial-tile work
retain dynamic scheduling. Other noncanonical G32 packing paths synchronize
once after writing their disjoint LHS regions.
Explicit SME/Neon partitioning uses the actual OpenMP team size. If the
runtime grants one worker, that worker computes both output partitions;
otherwise at least one worker remains assigned to each partition.

W8 calls with one row tile likewise cap the compute team at the number of
column tiles when that is smaller than the caller's thread pool. A single
column tile executes directly. Packing and multi-row-tile scheduling keep
their existing Torch parallel policy; nested calls do not create another
OpenMP team. This preserves the configured thread count and covers partial
column tiles even when the runtime grants fewer workers than requested.

GDN recurrence preserves the historical compiler contraction policy (GCC
`fast`, Clang/AppleClang `on`) for its multiply/add expressions. Native and
TLE Raw share the same recurrence body and FP32 state updates. SiLU packing
and the other sources keep `-ffp-contract=off`; applying that setting to GDN
would remove its native FMA instructions and change state rounding.
The GDN wrapper selects this convolution with
`FLAGGEMS_GDN_CONV_PREFILL_TRITON=1` (off unless requested).
The ordinary Triton width-four convolution retains the historical
`enable_fp_fusion=True` compilation policy. This is independent of the native
SiLU packing policy.
Its prepared programs use the historical static OpenMP distribution, with
the team bounded by the available programs and caller's thread setting.
Nested parallel calls execute serially. This keeps all workers useful when
the program count is not divisible by the thread count, including 128
programs on an 18-thread CPU.

Linear inputs are contiguous FP32 `[M,K]`; W4 G32 scales are BF16, W8 scales
are FP32.
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

This entry preserves positive input, state and weight strides; request metadata
and optional bias must be contiguous CPU tensors. It updates the first
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

## CPU Raw SiLU entries

SiLU warmup uses FlagTree-CPU's optional single-program C entry and an `i32`
status return. The ordinary Triton grid launcher keeps its original ABI.
Native and Raw use matched host validation, quantization and padded-panel
contracts. Invalid shapes, ABI mismatches and compilation failures raise errors;
there is no silent fallback to the native implementation.

By default Raw calls the standalone SiLU C ABI leaf. To link LLVM bitcode,
build with `FLAGGEMS_ARM_BITCODE_COMPILER=/path/to/matching/llvm/bin/clang++`
and set `FLAGGEMS_ARM_SILU_BITCODE=/path/to/flag_gems_arm_silu.bc` before
initialization. Match the FlagTree CPU LLVM version, OS, target features and
ABI; use PIC bitcode. Darwin builds also need the matching SDK sysroot.

`FLAGGEMS_ARM_SILU_BITCODE_MODE=kernel` (default) links arithmetic so LLVM can
inline it. `boundary` links only the exception boundary, which calls the
uniquely named `flag_gems_arm_silu_standalone_leaf` in the standalone library.
This preserves that library's arithmetic and scratch-storage ownership.
`FLAGGEMS_ARM_BITCODE_CPU=generic` is the portable default; `native` restricts
portability. Select a mode using measurements on the target platform.

Prepared entries disable dummy-pointer alignment specialization. Their DSOs
remain alive for the process lifetime; restart after updating a library or
bitcode. MR1/MR4 single-row entries and the general multi-row entry preserve
packing bytes, rounding and canonical zero padding. Native execution does not
require bitcode. `FLAGGEMS_ARM_SILU_SPECIALIZE_MR=0` disables MR specialization.
The public `pack_raw` schema and `pack.raw` alias select the same implementation.

## Ordinary Triton Gemma norm

`flag_gems.runtime.backend._arm.fused.gemma_norm.prepare_gemma_triton()`
prepares ordinary CPU Triton kernels once, then exposes
`gemma_rmsnorm.triton` and `gemma_fused_add_rmsnorm.triton` in the `gdn`
Torch namespace. Arithmetic runs in Triton; C++ only validates tensors,
allocates output, and launches cached entries through Torch's thread pool.
No Python launcher or JIT-cache lookup occurs on each inference call.

This specialization supports contiguous CPU BF16 input and weight, hidden
sizes 128/1024/2048, epsilon 1e-6, and at most INT32_MAX elements. Input/weight
views may have unaligned storage offsets. It preserves the native eight-lane
BF16 NEON accumulation and four-lane reduction order, plus BF16 residual
rounding before variance. Preparation checks the native vector layout and
uses its compiler contraction policy. Unsupported configurations should use
the existing native or CPU TLE overload rather than silently changing math.
The compiled libraries remain owned by the Python preparation module.

Short 128-wide Q/K normalization calls run serially to avoid a thread-team
fork that costs more than their arithmetic. Larger calls use the configured
Torch thread pool. CPU preparation and operator tests must use the same
FlagTree compiler, native library, and Torch build as inference.

## Reproducible optional installation

The Python wheel contains Python operators; it does not implicitly download
KleidiAI or manufacture platform DSOs. Build the optional extension from this
source checkout using the same Python/Torch installation as inference:

```sh
python -m pip install --no-build-isolation --no-deps .
python tools/build_arm_cpu.py --mode standalone \
  --build-dir build/arm-standalone --install-prefix "$VIRTUAL_ENV/lib/python3.11/site-packages"
```

Select the actual site-packages directory of your environment; Python 3.11 in
that example is not a version-independent path. Apple Clang additionally needs
`--openmp-include` pointing to your OpenMP header directory. For the optional
SiLU bitcode optimization, pass `--bitcode-compiler` from the LLVM toolchain
matching FlagTree-CPU. The DSO-only Raw route also works without this option.
Use a separate build directory for each mode. `--mode kai` additionally requires
explicit `--kai-root` and `--kai-library`; it is not required for standalone W8.

After loading a library with `arm_native_ops`, inference operations have
FakeTensor shape implementations, including their Raw overloads. Their mutation
and alias declarations remain in the C++ schemas. Weight packing, kernel-address
registration, and Raw warmup are eager initialization steps; initialize them
before capturing a graph. FakeTensor support alone does not certify every vLLM
compilation mode. W8 has opcheck and fullgraph aot_eager regression tests.

For dedicated AArch64 CI, set `FLAGGEMS_ARM_REQUIRE_NATIVE_TESTS=1` so a missing
library/compiler fails instead of silently skipping the operator suite. Set
`FLAGGEMS_ARM_NATIVE_LIBRARY` and `FLAGGEMS_ARM_SILU_LIBRARY` to the build artifacts
or install them into the tested package. Run the KAI and standalone suites in
separate processes, never load both namespace implementations in one process.
The standalone build excludes G32/MoE: select `-k 'standalone or gdn or gemma or
silu or conv or fake_schema or fake_mutation'` when running test_arm_native.py.
The KAI build runs the full suite, with the standalone-specific cases skipped.
SME2 cases may skip only when the host does not support that ISA.

## Tests

Use `pytest -s --tb=short tests/test_arm_native.py` after building the optional
libraries. The suite checks independent linear/quantization references,
MoE routing, mutable GDN state, native/Raw agreement, prepared Triton norm and
convolution, graph schemas and error recovery. Missing ISA features are skipped
explicitly. For missing-libdevice compatibility, use the existing
`tests/test_triton_lang_helper.py`.

Measure the native and selected Triton/Raw paths with matching inputs, shapes,
packing and thread counts on each target. Operator tests do not establish
whole-model speed or CIX P1 acceptance; keep benchmark evidence outside this
library's source tree.
