// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// CPU TLE Raw leaves. ATen vector/moments utilities preserve the reference
// Welford and BF16 rounding schedule; no Tensor dispatch occurs in these leaves.
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/Gelu.h>
#include <ATen/native/cpu/moments_utils.h>
#include <omp.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

using B = c10::BFloat16;
using BV = at::vec::Vectorized<B>;
using FV = at::vec::Vectorized<float>;

extern "C" void flag_gems_arm_add_layer_norm_bf16(uint64_t address) {
  const auto* d = reinterpret_cast<const int64_t*>(address);
  const auto* x = reinterpret_cast<const B*>(d[0]);
  const auto* r = reinterpret_cast<const B*>(d[1]);
  const auto* w = reinterpret_cast<const B*>(d[2]);
  const auto* b = reinterpret_cast<const B*>(d[3]);
  auto* y = reinterpret_cast<B*>(d[4]);
  const int64_t rows = d[5], n = d[6];
  const int threads = int(d[7]);
  const uint32_t bits = uint32_t(d[8]);
  float eps;
  std::memcpy(&eps, &bits, sizeof(eps));
#pragma omp parallel num_threads(threads)
  {
    alignas(64) B row[4096];
#pragma omp for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
      int64_t j = 0;
      for (; j + BV::size() <= n; j += BV::size())
        (BV::loadu(x + i * n + j) + BV::loadu(r + i * n + j)).store(row + j);
      for (; j < n; ++j) row[j] = B(float(x[i * n + j]) + float(r[i * n + j]));
      auto [mean, variance] = at::native::RowwiseMoments(row, n);
      const float scale = 1.0f / std::sqrt(variance + eps);
      const float shift = -scale * mean;
      j = 0;
      for (; j + BV::size() <= n; j += BV::size()) {
        auto [a0, a1] = at::vec::convert_to_float<B>(BV::loadu(row + j));
        auto [w0, w1] = at::vec::convert_to_float<B>(BV::loadu(w + j));
        auto [b0, b1] = at::vec::convert_to_float<B>(BV::loadu(b + j));
        auto y0 = (a0 * FV(scale) + FV(shift)) * w0 + b0;
        auto y1 = (a1 * FV(scale) + FV(shift)) * w1 + b1;
        at::vec::convert_from_float<B>(y0, y1).store(y + i * n + j);
      }
      for (; j < n; ++j) y[i * n + j] = B((float(row[j]) * scale + shift) * float(w[j]) + float(b[j]));
    }
  }
}

// Complete BF16 mapping of ATen's vector erf-GELU, without interpolation or
// approximation changes. ATen's scalar tails use a different erf algorithm;
// preserve their exact logical chunk boundaries even with fewer actual workers.
extern "C" void flag_gems_arm_gelu_lookup_bf16(uint64_t address) {
  static const auto table = [] {
    std::array<B, 65536> out;
    alignas(64) uint16_t bits[BV::size()];
    for (int i = 0; i < 65536; i += BV::size()) {
      for (int j = 0; j < BV::size(); ++j) bits[j] = uint16_t(i + j);
      at::native::vectorized_gelu<B>(BV::loadu(bits)).store(out.data() + i);
    }
    return out;
  }();
  const auto* d = reinterpret_cast<const int64_t*>(address);
  const auto* x = reinterpret_cast<const uint16_t*>(d[0]);
  auto* y = reinterpret_cast<B*>(d[1]);
  const int64_t n = d[2];
  const int threads = int(d[3]);
  const int64_t reference_threads = d[4];
  const int64_t grain = n > 16384 ? n / reference_threads : 32768;
  const int64_t tasks = n > grain ? std::min(reference_threads, (n + grain - 1) / grain) : 1;
  const int64_t chunk = (n + tasks - 1) / tasks;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t task = 0; task < tasks; ++task) {
    const int64_t begin = task * chunk, end = std::min(begin + chunk, n);
    const int64_t vector_end = begin + (end - begin) / (2 * BV::size()) * (2 * BV::size());
    for (int64_t j = begin; j < vector_end; ++j) y[j] = table[x[j]];
    for (int64_t j = vector_end; j < end; ++j) {
      B value;
      std::memcpy(&value, x + j, 2);
      y[j] = at::native::scalar_gelu<B>(value);
    }
  }
}
