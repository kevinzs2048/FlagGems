// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Match the installed PyTorch CPU vector math; preserve division then multiply.
#ifndef AT_BUILD_ARM_VEC256_WITH_SLEEF
#define AT_BUILD_ARM_VEC256_WITH_SLEEF 1
#endif
#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>

inline at::Tensor silu_mul(const at::Tensor& x) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat && x.is_contiguous() && x.dim() == 2 &&
                  x.size(1) > 0 && x.size(1) % 8 == 0,
              "Expected FP32 [M,2I], I aligned to 4");
  auto rows = x.size(0), inter = x.size(1) / 2;
  auto y = at::empty({rows, inter}, x.options());
  using Vec = at::vec::Vectorized<float>;
  const Vec one(1.f);
  for (int64_t row = 0; row < rows; ++row) {
    const float* gate = x.const_data_ptr<float>() + row * 2 * inter;
    const float* up = gate + inter;
    float* out = y.data_ptr<float>() + row * inter;
    for (int64_t c = 0; c < inter; c += Vec::size()) {
      auto g = Vec::loadu(gate + c), u = Vec::loadu(up + c);
      ((g / (one + g.neg().exp())) * u).store(out + c);
    }
  }
  return y;
}
