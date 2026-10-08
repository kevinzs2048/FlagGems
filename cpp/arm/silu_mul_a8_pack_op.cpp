// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <atomic>
#include <limits>
#include "silu_mul_a8_pack.h"

namespace {
// The optional CPU entry fixes grid=(1,1,1) and returns the final scalar
// output through the C ABI. No grid arguments or status pointer cross it.
using SiluRawKernel = int32_t (*)(const void*, void*, int32_t, int32_t, int32_t, int32_t);
int32_t uninitialized_silu_raw(const void*, void*, int32_t, int32_t, int32_t, int32_t) {
  TORCH_CHECK(false, "Initialize the SiLU CPU Raw entry first");
}
std::atomic<SiluRawKernel> raw_silu[3] {
    uninitialized_silu_raw, uninitialized_silu_raw, uninitialized_silu_raw};
at::Tensor silu_a8_impl(const at::Tensor& gu, int64_t mr, bool raw) {
  TORCH_CHECK(gu.device().is_cpu() && gu.scalar_type() == at::kFloat && gu.dim() == 2 && gu.size(1) > 0 &&
                  gu.size(1) % 64 == 0 && gu.stride(1) == 1 && (mr == 1 || mr == 4),
              "Expected CPU FP32 [M,2*K], K%32=0, inner stride1, MR1/4");
  TORCH_CHECK(gu.size(0) < std::numeric_limits<int32_t>::max() - 4 &&
                  gu.size(1) <= std::numeric_limits<int32_t>::max() &&
                  gu.stride(0) <= std::numeric_limits<int32_t>::max(),
              "Input exceeds int32 raw ABI");
  auto k = gu.size(1) / 2, m = gu.size(0);
  auto out = at::empty({(m + mr - 1) / mr * mr * (k + 8)}, gu.options().dtype(at::kByte));
  if (m && raw) {
    // Resolve Tensor metadata before the acquire that publishes JIT code.
    // These accesses do not depend on registration and must not serialize
    // behind its acquire barrier or keep the callable live across ATen calls.
    const auto* input = gu.const_data_ptr<float>();
    auto* output = out.data_ptr<uint8_t>();
    const int32_t stride = gu.stride(0);
    auto kernel = raw_silu[m == 1 ? (mr == 4) : 2].load(std::memory_order_acquire);
    const int32_t status =
        kernel(input, output, m, stride, k, mr);
    TORCH_CHECK(status == 0, "CPU Raw SiLU packing failed (status ", status, ")");
  } else if (m)
    TORCH_CHECK(flag_gems_arm_silu_mul_a8_pack(gu.const_data_ptr<float>(),
                                               out.data_ptr<uint8_t>(),
                                               m,
                                               gu.stride(0),
                                               k,
                                               mr) == 0,
                "Invalid raw SiLU pack dimensions");
  return out;
}
at::Tensor silu_a8(const at::Tensor& gu, int64_t mr) {
  return silu_a8_impl(gu, mr, false);
}
at::Tensor silu_a8_raw(const at::Tensor& gu, int64_t mr) {
  return silu_a8_impl(gu, mr, true);
}
}  // namespace
// Covers the scalar-return prototype and single-row/general registration slots.
extern "C" uint32_t flag_gems_arm_silu_raw_abi_version() noexcept { return 1; }
extern "C" void flag_gems_arm_register_silu_raw_kernel(uint64_t address) noexcept {
  for (auto& slot : raw_silu)
    slot.store(address ? reinterpret_cast<SiluRawKernel>(address) : uninitialized_silu_raw,
               std::memory_order_release);
}
extern "C" void flag_gems_arm_register_silu_raw_variant(uint32_t mr, uint64_t address) noexcept {
  if (mr == 0 || mr == 1 || mr == 4)
    raw_silu[mr == 0 ? 2 : (mr == 4)].store(
        address ? reinterpret_cast<SiluRawKernel>(address) : uninitialized_silu_raw,
        std::memory_order_release);
}
TORCH_LIBRARY(flag_gems_arm_silu_native, m) {
  m.def("pack(Tensor gate_up, int mr) -> Tensor");
  m.def("pack.raw(Tensor gate_up, int mr) -> Tensor");
  m.def("pack_raw(Tensor gate_up, int mr) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_silu_native, CPU, m) {
  m.impl("pack", silu_a8);
  m.impl("pack.raw", silu_a8_raw);
  m.impl("pack_raw", silu_a8_raw);
}
