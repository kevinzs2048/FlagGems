// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "cpu_features.h"
#include "raw_dispatch.h"
// W8 per-channel / dynamic asymmetric A8, using the existing KleidiAI library.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <kai/kai_common.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h>
#include <kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h>
#include <omp.h>
#include <torch/library.h>
#include <algorithm>
#include <limits>
namespace {
struct Kernel {
  size_t mr, nr, kr, sr, ms, ns;
  size_t (*lhs_offset)(size_t, size_t);
  size_t (*rhs_offset)(size_t, size_t);
  void (*run)(size_t, size_t, size_t, const void*, const void*, float*, size_t, size_t, float, float);
};
#define KERN(s)                                                                                              \
  Kernel {                                                                                                   \
    kai_get_mr_matmul_clamp_f32_##s(), kai_get_nr_matmul_clamp_f32_##s(), kai_get_kr_matmul_clamp_f32_##s(), \
        kai_get_sr_matmul_clamp_f32_##s(), kai_get_m_step_matmul_clamp_f32_##s(),                            \
        kai_get_n_step_matmul_clamp_f32_##s(), kai_get_lhs_packed_offset_matmul_clamp_f32_##s,               \
        kai_get_rhs_packed_offset_matmul_clamp_f32_##s, kai_run_matmul_clamp_f32_##s                         \
  }
Kernel kernel(bool sme, bool dot) {
  if (sme) {
    flag_gems::arm::check_matrix_feature(true, dot);
    if (dot) return KERN(qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot);
    return KERN(qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa);
  }
  if (dot) return KERN(qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod);
  return KERN(qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm);
}
at::Tensor pack(const at::Tensor& w, const at::Tensor& scales, bool sme) {
  TORCH_CHECK(w.device().is_cpu() && w.scalar_type() == at::kChar && w.dim() == 2 && w.is_contiguous(),
              "Expected contiguous INT8 [N,K]");
  auto n = w.size(0), k = w.size(1);
  TORCH_CHECK(n > 0 && k > 0 && k % 32 == 0, "Expected positive K multiple of 32");
  TORCH_CHECK(scales.device().is_cpu() && scales.scalar_type() == at::kFloat && scales.is_contiguous() &&
                  scales.numel() == n,
              "Expected FP32 channel scales");
  auto u = kernel(sme, false);
  auto size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, u.nr, u.kr, u.sr);
  auto out = at::empty({int64_t(size)}, w.options().dtype(at::kByte));
  kai_rhs_pack_qsi8cx_params params {1, 1.f};
  kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1,
                                           n,
                                           k,
                                           u.nr,
                                           u.kr,
                                           u.sr,
                                           w.const_data_ptr<int8_t>(),
                                           nullptr,
                                           scales.const_data_ptr<float>(),
                                           out.data_ptr(),
                                           0,
                                           &params);
  return out;
}
at::Tensor canonical(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2 && input.scalar_type() == at::kFloat &&
                  input.is_contiguous() && input.size(1) > 0 && input.size(1) % 32 == 0,
              "Expected FP32 [M,K], K multiple of 32");
  auto m = input.size(0), k = input.size(1);
  auto out = at::empty({m, k + 8}, input.options().dtype(at::kByte));
  at::parallel_for(0, m, 1, [&](int64_t a, int64_t b) {
    for (auto r = a; r < b; ++r)
      kai_run_lhs_quant_pack_qai8dxp_f32(1,
                                         k,
                                         1,
                                         8,
                                         1,
                                         0,
                                         input.const_data_ptr<float>() + r * k,
                                         k * 4,
                                         out.data_ptr<uint8_t>() + r * (k + 8));
  });
  return out;
}
struct DenseW8Plan {
  const float* input;
  const uint8_t* rhs;
  uint8_t* lhs;
  float* output;
  int64_t m, n, k, rows, cols;
  Kernel kernel;
  std::exception_ptr error;
};

}  // namespace
extern "C" void flag_gems_arm_w8_compute(uint64_t descriptor) noexcept {
  auto& plan = *reinterpret_cast<DenseW8Plan*>(descriptor);
  try {
    const auto& u = plan.kernel;
    const auto m = plan.m, n = plan.n, k = plan.k, rows = plan.rows, cols = plan.cols;
    at::parallel_for(0, (m + rows - 1) / rows, 1, [&](int64_t a, int64_t b) {
      for (auto j = a; j < b; ++j) {
        auto row = j * rows;
        kai_run_lhs_quant_pack_qai8dxp_f32(std::min(rows, m - row),
                                           k,
                                           u.mr,
                                           u.kr,
                                           u.sr,
                                           0,
                                           plan.input + row * k,
                                           k * 4,
                                           plan.lhs + u.lhs_offset(row, k));
      }
    });
    auto nc = (n + cols - 1) / cols;
    float inf = std::numeric_limits<float>::infinity();
    auto run_tiles = [&](int64_t a, int64_t b) {
      for (auto j = a; j < b; ++j) {
        auto row = (j / nc) * rows, col = (j % nc) * cols;
        u.run(std::min(rows, m - row),
              std::min(cols, n - col),
              k,
              plan.lhs + u.lhs_offset(row, k),
              plan.rhs + u.rhs_offset(col, k),
              plan.output + row * n + col,
              n * 4,
              4,
              -inf,
              inf);
      }
    };
    // A single row tile may expose fewer column jobs than the global pool.
    // The leaf touches only prepared buffers; use a bounded OpenMP team and
    // preserve the caller's thread policy and nested-parallel guard.
    if (m <= rows && nc < at::get_num_threads() && !at::in_parallel_region() && !omp_in_parallel()) {
      if (nc == 1) {
        run_tiles(0, 1);
      } else {
#pragma omp parallel for num_threads(nc) schedule(static)
        for (int64_t j = 0; j < nc; ++j) run_tiles(j, j + 1);
      }
    } else {
      at::parallel_for(0, ((m + rows - 1) / rows) * nc, 1, run_tiles);
    }
  } catch (...) {
    plan.error = std::current_exception();
  }
}
namespace {

__attribute__((noinline)) at::Tensor linear_impl(const at::Tensor& input,
                                                 const at::Tensor& packed,
                                                 int64_t n,
                                                 bool sme,
                                                 int64_t tile_rows,
                                                 int64_t tile_cols,
                                                 bool raw = false) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2 && input.scalar_type() == at::kFloat &&
                  input.is_contiguous(),
              "Expected contiguous FP32 [M,K]");
  auto m = input.size(0), k = input.size(1);
  TORCH_CHECK(m > 0 && n > 0 && k > 0 && k % 32 == 0 && tile_rows > 0 && tile_cols > 0, "Invalid dimensions");
  flag_gems::arm::check_matrix_feature(sme, m == 1);
  auto u = kernel(sme, m == 1);
  auto expected = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, u.nr, u.kr, u.sr);
  TORCH_CHECK(packed.device().is_cpu() && packed.scalar_type() == at::kByte && packed.is_contiguous() &&
                  packed.numel() == int64_t(expected),
              "Invalid packed weight");
  auto rows = m == 1 ? 1 : int64_t(kai_roundup(tile_rows, u.ms));
  auto cols = int64_t(kai_roundup(tile_cols, u.ns));
  auto lhs = at::empty({int64_t(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, u.mr, u.kr, u.sr))},
                       packed.options());
  auto out = at::empty({m, n}, input.options());
  DenseW8Plan plan {input.const_data_ptr<float>(),
                    packed.const_data_ptr<uint8_t>(),
                    lhs.data_ptr<uint8_t>(),
                    out.data_ptr<float>(),
                    m,
                    n,
                    k,
                    rows,
                    cols,
                    u,
                    {}};
  if (raw)
    flag_gems::arm::invoke_dense_compute(plan, 1);
  else
    flag_gems_arm_w8_compute(reinterpret_cast<uint64_t>(&plan));
  if (plan.error) std::rethrow_exception(plan.error);
  return out;
}
at::Tensor linear(const at::Tensor& input,
                  const at::Tensor& packed,
                  int64_t n,
                  bool sme,
                  int64_t tile_rows,
                  int64_t tile_cols) {
  return linear_impl(input, packed, n, sme, tile_rows, tile_cols);
}
at::Tensor canonical_raw(const at::Tensor& input) {
  return flag_gems::arm::invoke_raw([&] { return canonical(input); });
}
at::Tensor linear_raw(const at::Tensor& input,
                      const at::Tensor& packed,
                      int64_t n,
                      bool sme,
                      int64_t tile_rows,
                      int64_t tile_cols) {
  return linear_impl(input, packed, n, sme, tile_rows, tile_cols, true);
}

}  // namespace
TORCH_LIBRARY(flag_gems_arm_w8a8, m) {
  m.def("pack(Tensor weight, Tensor scales, bool sme) -> Tensor");
  m.def("canonical(Tensor input) -> Tensor");
  m.def("canonical.raw(Tensor input) -> Tensor");
  m.def("linear(Tensor input, Tensor packed, int n, bool sme, int tile_rows, int tile_cols) -> Tensor");
  m.def("linear.raw(Tensor input, Tensor packed, int n, bool sme, int tile_rows, int tile_cols) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_w8a8, CPU, m) {
  m.impl("pack", pack);
  m.impl("canonical", canonical);
  m.impl("canonical.raw", canonical_raw);
  m.impl("linear", linear);
  m.impl("linear.raw", linear_raw);
}
