// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
namespace flag_gems::arm {
enum class Feature { DotProd, I8mm, Sme2 };
inline bool has_feature(Feature feature) {
#if defined(__APPLE__)
  auto query = [](const char* name) {
    int value = 0;
    size_t length = sizeof(value);
    return sysctlbyname(name, &value, &length, nullptr, 0) == 0 && value;
  };
  static const bool dot = query("hw.optional.arm.FEAT_DotProd");
  static const bool i8mm = query("hw.optional.arm.FEAT_I8MM");
  static const bool sme = query("hw.optional.arm.FEAT_SME2");
#elif defined(__linux__)
  static const bool dot = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#ifdef HWCAP2_I8MM
  static const bool i8mm = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
  static const bool i8mm = false;
#endif
#ifdef HWCAP2_SME2
  static const bool sme = (getauxval(AT_HWCAP2) & HWCAP2_SME2) != 0;
#else
  static const bool sme = false;
#endif
#else
  static const bool dot = false, i8mm = false, sme = false;
#endif
  return feature == Feature::DotProd ? dot : (feature == Feature::I8mm ? i8mm : sme);
}
inline void check_matrix_feature(bool sme, bool dot) {
  TORCH_CHECK(has_feature(sme ? Feature::Sme2 : (dot ? Feature::DotProd : Feature::I8mm)),
              "Requested Arm KAI matrix ISA is unavailable on this CPU");
}
}  // namespace flag_gems::arm
