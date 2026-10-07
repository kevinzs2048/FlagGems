// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "silu_mul_a8_pack.h"
#include <arm_neon.h>
#include <cmath>
#include <cstring>
#include <vector>

// Same SLEEF-u10 vector exponential as installed ATen NEON FP32 SiLU.
extern "C" float32x4_t Sleef_expf4_u10(float32x4_t);

namespace {
// Exponential polynomial adapted from SLEEF (Boost Software License 1.0);
// see LICENSE.sleef.txt for the required license notice.
// Reuse the deployed FlagGems _sleef_expf_u10_inline polynomial, with explicit
// FMA only where SLEEF uses it. Keep library handling for extreme inputs.
inline float32x4_t exp_inline(float32x4_t x) {
  if (vmaxvq_f32(vabsq_f32(x)) > 80.f) return Sleef_expf4_u10(x);
  auto q = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
  auto qf = vcvtq_f32_s32(q);
  auto r = vfmaq_n_f32(x, qf, -0.693145751953125f);
  r = vfmaq_n_f32(r, qf, -1.428606765330187e-6f);
  auto p = vdupq_n_f32(0.00019852761761285365f);
  p = vfmaq_f32(vdupq_n_f32(0.0013930435525253415f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.008333360776305199f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.041666485369205475f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.1666666716337204f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  auto value = vaddq_f32(vdupq_n_f32(1), vfmaq_f32(r, vmulq_f32(r, r), p));
  auto half = vshrq_n_s32(q, 1);
  auto pow0 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(half, vdupq_n_s32(127)), 23));
  auto pow1 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vsubq_s32(q, half), vdupq_n_s32(127)), 23));
  return vmulq_f32(vmulq_f32(value, pow0), pow1);
}
}  // namespace

extern "C" int32_t flag_gems_arm_silu_mul_a8_pack(
    const float* gate_up, uint8_t* packed, int32_t m, int32_t stride, int32_t k, int32_t mr) {
  if (m < 0 || k <= 0 || k % 32 || stride < 0 || (mr != 1 && mr != 4)) return -1;
  if (m == 0) return 0;
  if (!gate_up || !packed) return -1;
  thread_local std::vector<float> scratch;
  if (scratch.size() < size_t(k)) scratch.resize(k);
  auto* act = scratch.data();
  const int32_t padded = (m + mr - 1) / mr * mr;
  for (int32_t row = 0; row < padded; ++row) {
    auto minv = vdupq_n_f32(0), maxv = vdupq_n_f32(0);
    if (row < m) {
      const float* gate = gate_up + int64_t(row) * stride;
      const float* up = gate + k;
      for (int32_t c = 0; c < k; c += 4) {
        auto g = vld1q_f32(gate + c), u = vld1q_f32(up + c);
        auto s = vdivq_f32(g, vaddq_f32(vdupq_n_f32(1), exp_inline(vnegq_f32(g))));
        auto value = vmulq_f32(s, u);
        vst1q_f32(act + c, value);
        minv = vminq_f32(minv, value);
        maxv = vmaxq_f32(maxv, value);
      }
    } else
      std::memset(act, 0, k * sizeof(float));
    const float lo = vminvq_f32(minv), hi = vmaxvq_f32(maxv);
    const float multiplier = lo == hi ? 1.f : 255.f / (hi - lo);
    const float scale = multiplier ? 1.f / multiplier : 0.f;
    const float dlo = lo * multiplier, dhi = hi * multiplier;
    float zp = ((-128.f + dlo) + (127.f + dhi) > 0.f) ? -128.f - dlo : 127.f - dhi;
    zp = std::fmin(std::fmax(zp, -128.f), 127.f);
    const int32_t izp = int32_t(std::nearbyint(zp));
    const int32_t panel = row / mr, lane = row % mr;
    uint8_t* base = packed + int64_t(panel) * mr * (k + 8);
    const auto bias = vdupq_n_s32(izp), qmin = vdupq_n_s32(-128), qmax = vdupq_n_s32(127);
    for (int32_t c = 0; c < k; c += 8) {
      auto q0 = vaddq_s32(vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(act + c), multiplier)), bias);
      auto q1 = vaddq_s32(vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(act + c + 4), multiplier)), bias);
      q0 = vminq_s32(vmaxq_s32(q0, qmin), qmax);
      q1 = vminq_s32(vmaxq_s32(q1, qmin), qmax);
      vst1_s8(reinterpret_cast<int8_t*>(base + c * mr + lane * 8),
              vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))));
    }
    const int32_t offset = -izp;
    std::memcpy(base + mr * k + lane * 4, &offset, 4);
    std::memcpy(base + mr * k + mr * 4 + lane * 4, &scale, 4);
  }
  return 0;
}

// Scalar CPU extern ABI: addresses travel as uint64_t because this Triton
// version cannot use pointer_type keys in extern_elementwise dispatch maps.
extern "C" int32_t flag_gems_arm_silu_mul_a8_pack_addr(
    uint64_t x, uint64_t out, int32_t m, int32_t stride, int32_t k, int32_t mr) {
  try {
    return flag_gems_arm_silu_mul_a8_pack(reinterpret_cast<const float*>(uintptr_t(x)),
                                          reinterpret_cast<uint8_t*>(uintptr_t(out)),
                                          m,
                                          stride,
                                          k,
                                          mr);
  } catch (...) {
    // Allocation failures must not unwind through a JIT frame.
    return -2;
  }
}
