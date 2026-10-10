// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "raw_dispatch.h"

namespace flag_gems::arm {
std::atomic<RawKernel> raw_kernel {nullptr};
std::atomic<RawKernel> dense_raw_kernels[3] {};
}  // namespace flag_gems::arm

extern "C" void flag_gems_arm_register_raw_kernel(uint64_t address) noexcept {
  flag_gems::arm::raw_kernel.store(reinterpret_cast<flag_gems::arm::RawKernel>(address),
                                   std::memory_order_release);
}

extern "C" void flag_gems_arm_raw_dispatch(uint64_t descriptor) noexcept {
  // Only the synchronous C++ caller creates this descriptor. The callback
  // catches C++ exceptions before returning across the compiled C ABI.
  auto& context = *reinterpret_cast<flag_gems::arm::RawInvocation*>(descriptor);
  context.invoke(&context);
}

extern "C" void flag_gems_arm_register_dense_raw_kernel(uint32_t index, uint64_t address) noexcept {
  if (index < 3)
    flag_gems::arm::dense_raw_kernels[index].store(reinterpret_cast<flag_gems::arm::RawKernel>(address),
                                                   std::memory_order_release);
}
