# Optional Arm CPU operators

This target keeps Arm inference arithmetic in FlagGems. It provides W4A8 G32
and per-channel W8A8 linear operators, routed MoE, fused SiLU/up/A8 packing,
and gated delta network (GDN) recurrence/convolution/normalization operators.
Both Darwin AArch64 and Linux AArch64 are supported. The target is opt-in and
does not change the default dispatcher or existing GPU builds.

## Build

Use the same Python and Torch environment as inference. The dense kernels
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
void returns (the `triton_v3.7.x` / 3.7.2 native-call port). A CPU build lacking
that optional capability can still import FlagGems and use native operators.

The chain is a compiled `@triton.jit` entry, CPU `raw_call`, then C++/Neon/KAI.
It is a CPU native-call API, not source-region embedding. Warmup retains each
compiled DSO and registers its compute address once. C++ calls that address
synchronously; inference does not use a Python Triton launcher, per-expert
lookup, or per-call JIT lookup. Small dense GEMMs have typed operation entries;
composite MoE/GDN operations cross the boundary once per complete operation.
The C ABI catches C++ exceptions and the Torch caller rethrows them after the
compiled entry returns. Tensor references and descriptors stay alive until
all parallel work completes.

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

### Candidate acceptance status (2026-10-08)

These are opt-in candidate APIs. Functional coverage does not imply that every
shape or platform meets the performance gate. The latest Darwin test run has
61 passing cases. The last Linux AArch64 run before adding the decode wrapper
has 36 passing cases and 9 SME2 cases skipped on unsupported hardware. The
decode wrapper and its additional cases still require Linux revalidation.

Paired Darwin public-entry measurements of width-four BF16 decode convolution
at batch 1 (including cached metadata and Python transposes) give:

| Channels | Threads | Native (us) | Triton (us) | Paired latency change |
| --- | --- | --- | --- | --- |
| 6144 | 8 | 16.48 | 10.30 | -37.26% |
| 8192 | 8 | 21.02 | 11.94 | -42.94% |
| 6144 | 18 | 10.03 | 9.59 | -5.16% |
| 8192 | 18 | 12.31 | 10.43 | -15.21% |

SiLU/A8's latest Darwin direct Torch-entry comparison covers rows 1, 4, 7,
16, 32 and 512 at K=512 and MR=1/4: paired median Raw latency changes range
from -0.04% to +0.33%. Other tested K values and empty rows have correctness
coverage. Public Python-entry savings also include loader overhead differences
and must not be described as improvements to native arithmetic itself.

Some small dense Raw configurations remain unaccepted: the latest Linux
diagnostic has median regressions of approximately 0.58% to 2.39%; one Darwin
G32 case has approximately 0.91%. Confidence intervals crossing 0.5% also
remain unresolved. Retain direct native execution for configurations without
validated acceptance; Raw is not automatically enabled by this patch. Whole
model PP512/TG128 and model routing have not been revalidated against these
candidate libraries. The optional APIs are suitable for review and further
measurement, not a claim that all upstream acceptance work is complete.
