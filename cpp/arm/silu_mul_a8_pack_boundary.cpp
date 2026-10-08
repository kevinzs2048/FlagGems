// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Link the exception boundary into a CPU TLE entry; retain native arithmetic.
#include <cstdint>
extern "C" int32_t flag_gems_arm_silu_standalone_leaf(
    const float*, uint8_t*, int32_t, int32_t, int32_t, int32_t);
extern "C" int32_t flag_gems_arm_silu_mul_a8_pack_safe(
    uint64_t x, uint64_t out, int32_t m, int32_t stride,
    int32_t k, int32_t mr) noexcept {
  try {
    return flag_gems_arm_silu_standalone_leaf(
        reinterpret_cast<const float*>(x), reinterpret_cast<uint8_t*>(out),
        m, stride, k, mr);
  } catch (...) {
    return -2;
  }
}
