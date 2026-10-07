// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Internal, version-specific interface to the installed Torch KAI symbols.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <arm_neon.h>
#include <dlfcn.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#include <kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h>
#include <cstring>
#include <limits>
#include "cpu_features.h"
namespace flag_gems_runtime_kai {
// Runtime-packed expert weights belong to Torch's KAI ABI. Resolve its exact
// implementation explicitly; a separately linked KAI library for dense SME
// kernels must not silently replace expert quantization or arithmetic.
template <class F>
F torch_kai(const char* symbol) {
  static void* handle = [] {
    Dl_info info {};
    TORCH_CHECK(dladdr(reinterpret_cast<void*>(&at::get_num_threads), &info) && info.dli_fname,
                "Cannot locate the Torch CPU operator library");
    auto library = dlopen(info.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    TORCH_CHECK(library, "Cannot retain the Torch CPU operator library");
    return library;
  }();
  auto address = dlsym(handle, symbol);
  TORCH_CHECK(address, "Torch does not export the required KAI ABI symbol: ", symbol);
  return reinterpret_cast<F>(address);
}
#define TORCH_KAI(fn) torch_kai<decltype(&fn)>(#fn)
struct Kernel {
  size_t mr, kr, sr;
  size_t (*rhs_offset)(size_t, size_t, size_t);
  void (*run)(size_t, size_t, size_t, size_t, const void*, const void*, float*, size_t, size_t, float, float);
};
#define KERNEL(s)                                                                                          \
  Kernel {                                                                                                 \
    TORCH_KAI(kai_get_mr_matmul_clamp_f32_##s)(), TORCH_KAI(kai_get_kr_matmul_clamp_f32_##s)(),            \
        TORCH_KAI(kai_get_sr_matmul_clamp_f32_##s)(),                                                      \
        TORCH_KAI(kai_get_rhs_packed_offset_matmul_clamp_f32_##s), TORCH_KAI(kai_run_matmul_clamp_f32_##s) \
  }
inline Kernel kernel(int64_t rows) {
  flag_gems::arm::check_matrix_feature(false, rows == 1);
  static const Kernel dot = KERNEL(qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod);
  static const Kernel matrix = KERNEL(qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm);
  return rows == 1 ? dot : matrix;
}
inline void check_weights(const at::Tensor& w, int64_t experts, int64_t n, int64_t k) {
  auto u = kernel(4);
  TORCH_CHECK(w.device().is_cpu() && w.scalar_type() == at::kByte && w.is_contiguous() && w.dim() == 2 &&
                  w.size(0) == experts && w.size(1) == int64_t(u.rhs_offset(n, k, 32)),
              "Expected Runtime KAI G32 packed weights [E,bytes]");
}
inline at::Tensor quantize(const at::Tensor& x) {
  static const auto pack = TORCH_KAI(kai_run_lhs_quant_pack_qai8dxp_f32);
  auto m = x.size(0), k = x.size(1);
  auto q = at::empty({m, k + 8}, x.options().dtype(at::kByte));
  at::parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    for (auto row = begin; row < end; ++row)
      pack(1,
           k,
           1,
           8,
           1,
           0,
           x.const_data_ptr<float>() + row * k,
           k * 4,
           q.data_ptr<uint8_t>() + row * (k + 8));
  });
  return q;
}
inline at::Tensor packed_gemm(const at::Tensor& canonical,
                              const int64_t* token_ids,
                              int64_t count,
                              const uint8_t* rhs,
                              int64_t n,
                              bool blocked_pack = false) {
  const int64_t k = canonical.size(1) - 8;
  auto u = kernel(count);
  TORCH_CHECK(u.kr / u.sr == 8, "Unsupported installed KAI LHS layout");
  auto lhs =
      at::empty({int64_t(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(count, k, u.mr, u.kr, u.sr))},
                canonical.options());
  auto* dst = lhs.data_ptr<uint8_t>();
  const auto* src = canonical.const_data_ptr<uint8_t>();
  int64_t first_row = 0;
  if (blocked_pack && u.mr == 1) {
    std::memcpy(dst, src + token_ids[0] * (k + 8), k + 8);
    first_row = 1;
  } else if (blocked_pack && u.mr == 4) {
    // Transpose four canonical rows into contiguous 8-byte KAI lanes.
    // Retain the Runtime's per-row A8 codes, offset and multiplier exactly.
    for (; first_row + 4 <= count; first_row += 4) {
      const uint8_t* rows[4];
      for (int r = 0; r < 4; ++r) rows[r] = src + token_ids[first_row + r] * (k + 8);
      auto* block = dst + first_row * (k + 8);
      for (int64_t c = 0; c < k; c += 16) {
        auto a = vreinterpretq_u64_u8(vld1q_u8(rows[0] + c));
        auto b = vreinterpretq_u64_u8(vld1q_u8(rows[1] + c));
        auto d = vreinterpretq_u64_u8(vld1q_u8(rows[2] + c));
        auto e = vreinterpretq_u64_u8(vld1q_u8(rows[3] + c));
        vst1q_u8(block + 4 * c, vreinterpretq_u8_u64(vzip1q_u64(a, b)));
        vst1q_u8(block + 4 * c + 16, vreinterpretq_u8_u64(vzip1q_u64(d, e)));
        vst1q_u8(block + 4 * c + 32, vreinterpretq_u8_u64(vzip2q_u64(a, b)));
        vst1q_u8(block + 4 * c + 48, vreinterpretq_u8_u64(vzip2q_u64(d, e)));
      }
      for (int r = 0; r < 4; ++r) {
        std::memcpy(block + 4 * k + r * 4, rows[r] + k, 4);
        std::memcpy(block + 4 * (k + 4) + r * 4, rows[r] + k + 4, 4);
      }
    }
  }
  for (int64_t row = first_row; row < count; ++row) {
    const auto* from = src + token_ids[row] * (k + 8);
    auto* block = dst + (row / u.mr) * u.mr * (k + 8);
    auto slot = row % u.mr;
    for (int64_t c = 0; c < k; c += 8) std::memcpy(block + (c / 8 * u.mr + slot) * 8, from + c, 8);
    std::memcpy(block + u.mr * k + slot * 4, from + k, 4);
    std::memcpy(block + u.mr * (k + 4) + slot * 4, from + k + 4, 4);
  }
  auto out = at::empty({count, n}, canonical.options().dtype(at::kFloat));
  const float inf = std::numeric_limits<float>::infinity();
  u.run(count, n, k, 32, dst, rhs, out.data_ptr<float>(), n * 4, 4, -inf, inf);
  return out;
}

#undef KERNEL
}  // namespace flag_gems_runtime_kai
