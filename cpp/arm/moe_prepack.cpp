// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "raw_dispatch.h"
// Quantize each input token once, gather packed A8 rows per expert.
// All KAI symbols resolve to the installed libtorch_cpu (no static KAI library).
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/_dyn_quant_matmul_4bit.h>
#include <arm_neon.h>
#include <c10/core/InferenceMode.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#include <kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h>
#include <omp.h>
#include <torch/library.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <tuple>
#include <vector>
#include "runtime_kai.h"
#include "silu_mul_a8_pack.h"

namespace {
using namespace flag_gems_runtime_kai;
at::Tensor quantize_input(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2 && input.size(0) > 0 && input.size(1) > 0 &&
                  input.size(1) % 32 == 0 &&
                  (input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16),
              "Expected CPU FP32/BF16 G32 input");
  return quantize(input.to(at::kFloat).contiguous());
}
at::Tensor linear(const at::Tensor& input, const at::Tensor& weight, int64_t n) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2 && input.scalar_type() == at::kFloat &&
                  input.size(0) > 0 && input.size(1) > 0 && input.size(1) % 32 == 0 && n > 0 && n % 8 == 0,
              "Invalid G32 input");
  auto x = input.contiguous();
  check_weights(weight, 1, n, x.size(1));
  auto q = quantize(x);
  std::vector<int64_t> ids(x.size(0));
  for (int64_t i = 0; i < x.size(0); ++i) ids[i] = i;
  return packed_gemm(q, ids.data(), x.size(0), weight.const_data_ptr<uint8_t>(), n);
}
struct MoEPartial {
  at::Tensor partial;
  std::vector<int64_t> positions;
  int64_t tokens, top;
};
MoEPartial compute_parts(const at::Tensor& input,
                         const at::Tensor& logits,
                         const at::Tensor& w1,
                         const at::Tensor& w2,
                         int64_t h,
                         int64_t inter,
                         int64_t top,
                         bool renormalize,
                         bool blocked_pack = false,
                         bool fused_activation = false) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2 && input.size(1) == h &&
                  (input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16) &&
                  input.size(0) > 0 && h > 0 && inter > 0 && h % 32 == 0 && inter % 32 == 0,
              "Invalid MoE input");
  TORCH_CHECK(logits.device().is_cpu() && logits.dim() == 2 && logits.size(0) == input.size(0) &&
                  (logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16) && top > 0 &&
                  top <= logits.size(1),
              "Invalid router logits");
  const int64_t tokens = input.size(0), experts = logits.size(1), routes = tokens * top;
  check_weights(w1, experts, 2 * inter, h);
  check_weights(w2, experts, h, inter);
  auto selected = at::topk(logits, top, -1, true, false);
  auto values = std::get<0>(selected), ids = std::get<1>(selected).contiguous();
  auto gates = (renormalize ? at::softmax(values, -1) : at::exp(values - at::logsumexp(logits, {-1}, true)))
                   .to(at::kFloat)
                   .contiguous();
  auto q = quantize(input.to(at::kFloat).contiguous());
  const auto* id = ids.const_data_ptr<int64_t>();
  const auto* g = gates.const_data_ptr<float>();
  std::vector<int64_t> offsets(experts + 1, 0), cursor(experts, 0), positions(routes), token_ids(routes);
  std::vector<float> expert_gates(routes);
  for (int64_t r = 0; r < routes; ++r) ++offsets[id[r] + 1];
  for (int64_t e = 0; e < experts; ++e) offsets[e + 1] += offsets[e];
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t r = 0; r < top; ++r) {
      auto idx = t * top + r, e = id[idx], p = offsets[e] + cursor[e]++;
      token_ids[p] = t;
      expert_gates[p] = g[idx];
      positions[idx] = p;
    }
    std::sort(positions.begin() + t * top, positions.begin() + (t + 1) * top);
  }
  auto partial = at::empty({routes, h}, input.options().dtype(at::kFloat));
  auto run = [&](int64_t e, int64_t start, int64_t count) {
    auto gu = packed_gemm(q,
                          token_ids.data() + start,
                          count,
                          w1.const_data_ptr<uint8_t>() + e * w1.size(1),
                          2 * inter,
                          blocked_pack);
    if (fused_activation) {
      auto u = kernel(count);
      auto packed = at::empty(
          {int64_t(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(count, inter, u.mr, u.kr, u.sr))},
          q.options());
      TORCH_CHECK(flag_gems_arm_silu_mul_a8_pack(gu.const_data_ptr<float>(),
                                                 packed.data_ptr<uint8_t>(),
                                                 count,
                                                 gu.stride(0),
                                                 inter,
                                                 u.mr) == 0,
                  "Fused SiLU/A8 packing failed");
      const float inf = std::numeric_limits<float>::infinity();
      auto* dst = partial.data_ptr<float>() + start * h;
      u.run(count,
            h,
            inter,
            32,
            packed.const_data_ptr<uint8_t>(),
            w2.const_data_ptr<uint8_t>() + e * w2.size(1),
            dst,
            h * 4,
            4,
            -inf,
            inf);
      for (int64_t row = 0; row < count; ++row) {
        auto gate = vdupq_n_f32(expert_gates[start + row]);
        for (int64_t c = 0; c < h; c += 4)
          vst1q_f32(dst + row * h + c, vmulq_f32(vld1q_f32(dst + row * h + c), gate));
      }
    } else {
      auto act = at::silu(gu.narrow(1, 0, inter)).mul_(gu.narrow(1, inter, inter));
      auto y = at::_ops::_dyn_quant_matmul_4bit::call(act, w2[e], 32, inter, h);
      auto* dst = partial.data_ptr<float>() + start * h;
      const auto* src = y.const_data_ptr<float>();
      for (int64_t row = 0; row < count; ++row) {
        auto gate = vdupq_n_f32(expert_gates[start + row]);
        for (int64_t c = 0; c < h; c += 4)
          vst1q_f32(dst + row * h + c, vmulq_f32(vld1q_f32(src + row * h + c), gate));
      }
    }
  };
  int team = (at::in_parallel_region() || omp_in_parallel()) ? 1 : at::get_num_threads();
  int64_t active = 0;
  for (int64_t e = 0; e < experts; ++e) active += offsets[e + 1] > offsets[e];
  if (active < 4 * team) {
    at::parallel_for(0, routes, 0, [&](int64_t begin, int64_t end) {
      c10::InferenceMode guard;
      auto first = std::upper_bound(offsets.begin(), offsets.end(), begin) - offsets.begin() - 1;
      for (auto e = first; e < experts && offsets[e] < end; ++e) {
        auto start = std::max(offsets[e], begin), count = std::min(offsets[e + 1], end) - start;
        if (count > 0) run(e, start, count);
      }
    });
  } else {
    struct Job {
      int64_t e, start, count;
    };
    std::vector<Job> jobs;
    std::atomic_flag failed = ATOMIC_FLAG_INIT;
    std::exception_ptr error;
#pragma omp parallel num_threads(team)
    {
      c10::InferenceMode inference_guard;
      c10::ParallelGuard parallel_guard(true);
      at::internal::ThreadIdGuard thread_guard(omp_get_thread_num());
#pragma omp single
      {
        try {
          auto chunk = (routes + omp_get_num_threads() - 1) / omp_get_num_threads();
          for (int64_t begin = 0; begin < routes; begin += chunk) {
            auto end = std::min(routes, begin + chunk);
            auto first = std::upper_bound(offsets.begin(), offsets.end(), begin) - offsets.begin() - 1;
            for (auto e = first; e < experts && offsets[e] < end; ++e) {
              auto start = std::max(offsets[e], begin), count = std::min(offsets[e + 1], end) - start;
              if (count > 0) jobs.push_back({e, start, count});
            }
          }
          std::stable_sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.count > b.count;
          });
        } catch (...) {
          failed.test_and_set();
          error = std::current_exception();
        }
      }
#pragma omp for schedule(dynamic, 1)
      for (size_t j = 0; j < jobs.size(); ++j) {
        try {
          auto job = jobs[j];
          run(job.e, job.start, job.count);
        } catch (...) {
          if (!failed.test_and_set()) error = std::current_exception();
        }
      }
    }
    if (error) std::rethrow_exception(error);
  }
  return {partial, std::move(positions), tokens, top};
}
at::Tensor moe_impl(const at::Tensor& input,
                    const at::Tensor& logits,
                    const at::Tensor& w1,
                    const at::Tensor& w2,
                    int64_t h,
                    int64_t inter,
                    int64_t top,
                    bool renormalize,
                    bool blocked_pack,
                    bool fused_activation = false) {
  auto parts =
      compute_parts(input, logits, w1, w2, h, inter, top, renormalize, blocked_pack, fused_activation);
  auto& partial = parts.partial;
  auto& positions = parts.positions;
  auto tokens = parts.tokens;
  auto output = at::empty({tokens, h}, partial.options());
  const float* p = partial.const_data_ptr<float>();
  auto* out = output.data_ptr<float>();
  at::parallel_for(0, tokens, 1, [&](int64_t begin, int64_t end) {
    for (auto t = begin; t < end; ++t)
      for (int64_t c = 0; c < h; c += 4) {
        auto sum = vdupq_n_f32(0);
        for (int64_t r = 0; r < top; ++r) sum = vaddq_f32(sum, vld1q_f32(p + positions[t * top + r] * h + c));
        vst1q_f32(out + t * h + c, sum);
      }
  });
  return output.to(input.scalar_type());
}
at::Tensor moe(const at::Tensor& x,
               const at::Tensor& r,
               const at::Tensor& w1,
               const at::Tensor& w2,
               int64_t h,
               int64_t i,
               int64_t top,
               bool norm) {
  return moe_impl(x, r, w1, w2, h, i, top, norm, false);
}
at::Tensor moe_blocked(const at::Tensor& x,
                       const at::Tensor& r,
                       const at::Tensor& w1,
                       const at::Tensor& w2,
                       int64_t h,
                       int64_t i,
                       int64_t top,
                       bool norm) {
  return moe_impl(x, r, w1, w2, h, i, top, norm, true);
}
std::tuple<at::Tensor, at::Tensor> moe_parts(const at::Tensor& input,
                                             const at::Tensor& logits,
                                             const at::Tensor& w1,
                                             const at::Tensor& w2,
                                             int64_t h,
                                             int64_t inter,
                                             int64_t top,
                                             bool renormalize) {
  auto parts = compute_parts(input, logits, w1, w2, h, inter, top, renormalize);
  auto positions = at::empty({parts.tokens, parts.top}, input.options().dtype(at::kLong));
  std::memcpy(positions.data_ptr<int64_t>(),
              parts.positions.data(),
              parts.positions.size() * sizeof(int64_t));
  return {parts.partial, positions};
}
at::Tensor quantize_input_raw(const at::Tensor& input) {
  return flag_gems::arm::invoke_raw([&] { return quantize_input(input); });
}
at::Tensor linear_raw(const at::Tensor& input, const at::Tensor& weight, int64_t n) {
  return flag_gems::arm::invoke_raw([&] { return linear(input, weight, n); });
}
__attribute__((noinline)) at::Tensor moe_fused(const at::Tensor& x,
                     const at::Tensor& r,
                     const at::Tensor& w1,
                     const at::Tensor& w2,
                     int64_t h,
                     int64_t i,
                     int64_t top,
                     bool norm) {
  return moe_impl(x, r, w1, w2, h, i, top, norm, false, true);
}
at::Tensor moe_fused_raw(const at::Tensor& x,
                         const at::Tensor& r,
                         const at::Tensor& w1,
                         const at::Tensor& w2,
                         int64_t h,
                         int64_t i,
                         int64_t top,
                         bool norm) {
  return flag_gems::arm::invoke_raw([&] { return moe_fused(x, r, w1, w2, h, i, top, norm); });
}
at::Tensor moe_raw(const at::Tensor& x,
                   const at::Tensor& r,
                   const at::Tensor& w1,
                   const at::Tensor& w2,
                   int64_t h,
                   int64_t i,
                   int64_t top,
                   bool norm) {
  return flag_gems::arm::invoke_raw([&] { return moe(x, r, w1, w2, h, i, top, norm); });
}
at::Tensor moe_blocked_raw(const at::Tensor& x,
                           const at::Tensor& r,
                           const at::Tensor& w1,
                           const at::Tensor& w2,
                           int64_t h,
                           int64_t i,
                           int64_t top,
                           bool norm) {
  return flag_gems::arm::invoke_raw([&] { return moe_blocked(x, r, w1, w2, h, i, top, norm); });
}

}  // namespace
TORCH_LIBRARY(flag_gems_arm_moe_prepack, m) {
  m.def("quantize_input(Tensor input) -> Tensor");
  m.def("quantize_input.raw(Tensor input) -> Tensor");
  m.def("linear(Tensor input, Tensor weight, int n) -> Tensor");
  m.def("linear.raw(Tensor input, Tensor weight, int n) -> Tensor");
  m.def(
      "moe(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool renormalize) "
      "-> Tensor");
  m.def(
      "moe_fused(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_fused.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_blocked(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_blocked.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_parts(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_moe_prepack, CPU, m) {
  m.impl("linear", linear);
  m.impl("linear.raw", linear_raw);
  m.impl("moe", moe);
  m.impl("moe_fused", moe_fused);
  m.impl("moe_fused.raw", moe_fused_raw);
  m.impl("moe.raw", moe_raw);
  m.impl("moe_blocked", moe_blocked);
  m.impl("moe_blocked.raw", moe_blocked_raw);
  m.impl("moe_parts", moe_parts);
  m.impl("quantize_input", quantize_input);
  m.impl("quantize_input.raw", quantize_input_raw);
}
