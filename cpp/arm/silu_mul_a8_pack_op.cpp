// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <atomic>
#include <limits>
#include "silu_mul_a8_pack.h"

namespace {
using SiluRawKernel = void (*)(const void*,
                               void*,
                               int32_t,
                               int32_t,
                               int32_t,
                               int32_t,
                               int32_t*,
                               uint32_t,
                               uint32_t,
                               uint32_t,
                               uint32_t,
                               uint32_t,
                               uint32_t);
std::atomic<SiluRawKernel> raw_silu {nullptr};
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
    auto kernel = raw_silu.load(std::memory_order_acquire);
    TORCH_CHECK(kernel, "Initialize the SiLU CPU Raw entry first");
    int32_t status = 0;
    kernel(gu.const_data_ptr<float>(),
           out.data_ptr<uint8_t>(),
           m,
           gu.stride(0),
           k,
           mr,
           &status,
           0,
           0,
           0,
           1,
           1,
           1);
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
extern "C" void flag_gems_arm_register_silu_raw_kernel(uint64_t address) noexcept {
  raw_silu.store(reinterpret_cast<SiluRawKernel>(address), std::memory_order_release);
}
TORCH_LIBRARY(flag_gems_arm_silu_native, m) {
  m.def("pack(Tensor gate_up, int mr) -> Tensor");
  m.def("pack.raw(Tensor gate_up, int mr) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_silu_native, CPU, m) {
  m.impl("pack", silu_a8);
  m.impl("pack.raw", silu_a8_raw);
}
