// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "raw_dispatch.h"
// Keep quantized intermediate activations; materialize only one output stripe.
#include <arm_neon.h>
#include <c10/core/InferenceMode.h>
#include <omp.h>
#include <torch/library.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <vector>
#include "runtime_kai.h"

namespace {
using namespace flag_gems_runtime_kai;
at::Tensor streamed(const at::Tensor& input,
                    const at::Tensor& logits,
                    const at::Tensor& w1,
                    const at::Tensor& w2,
                    int64_t h,
                    int64_t inter,
                    int64_t top,
                    bool renormalize,
                    int64_t tile_n) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2 && input.size(1) == h &&
                  (input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16) &&
                  input.size(0) > 0 && h > 0 && inter > 0 && h % 32 == 0 && inter % 32 == 0,
              "Invalid MoE input");
  TORCH_CHECK(logits.device().is_cpu() && logits.dim() == 2 && logits.size(0) == input.size(0) &&
                  (logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16) && top > 0 &&
                  top <= logits.size(1),
              "Invalid router logits");
  TORCH_CHECK(tile_n >= 32 && tile_n <= 2048 && tile_n % 32 == 0,
              "Output tile must be a multiple of 32 in [32,2048]");
  const int64_t tokens = input.size(0), experts = logits.size(1), routes = tokens * top;
  const int64_t stripe = std::min(tile_n, h);
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
  const int team = (at::in_parallel_region() || omp_in_parallel()) ? 1 : at::get_num_threads();
  struct Job {
    int64_t e, start, count, pack_offset;
  };
  std::vector<Job> jobs;
  const auto chunk = (routes + team - 1) / team;
  int64_t pack_bytes = 0;
  for (int64_t begin = 0; begin < routes; begin += chunk) {
    auto end = std::min(routes, begin + chunk);
    auto first = std::upper_bound(offsets.begin(), offsets.end(), begin) - offsets.begin() - 1;
    for (auto e = first; e < experts && offsets[e] < end; ++e) {
      auto start = std::max(offsets[e], begin), count = std::min(offsets[e + 1], end) - start;
      if (count <= 0) continue;
      auto u = kernel(count);
      jobs.push_back({e, start, count, pack_bytes});
      pack_bytes += kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(count, inter, u.mr, u.kr, u.sr);
    }
  }
  std::stable_sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.count > b.count; });
  auto activation_pack = at::empty({pack_bytes}, q.options());
  auto partial = at::empty({routes, stripe}, input.options().dtype(at::kFloat));
  auto output = at::empty({tokens, h}, partial.options());
  const float inf = std::numeric_limits<float>::infinity();
  std::atomic<bool> failed {false};
  std::exception_ptr error;
  auto fail = [&]() {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true)) error = std::current_exception();
  };
#pragma omp parallel num_threads(team)
  {
    c10::InferenceMode inference_guard;
    c10::ParallelGuard parallel_guard(true);
    at::internal::ThreadIdGuard thread_guard(omp_get_thread_num());
// Each job's activated A8 rows are packed once and reused across stripes.
#pragma omp for schedule(dynamic, 1)
    for (size_t j = 0; j < jobs.size(); ++j) {
      if (failed.load(std::memory_order_relaxed)) continue;
      try {
        auto job = jobs[j];
        auto u = kernel(job.count);
        auto gu = packed_gemm(q,
                              token_ids.data() + job.start,
                              job.count,
                              w1.const_data_ptr<uint8_t>() + job.e * w1.size(1),
                              2 * inter);
        auto act = at::silu(gu.narrow(1, 0, inter)).mul_(gu.narrow(1, inter, inter));
        kai_run_lhs_quant_pack_qai8dxp_f32(job.count,
                                           inter,
                                           u.mr,
                                           u.kr,
                                           u.sr,
                                           0,
                                           act.const_data_ptr<float>(),
                                           inter * 4,
                                           activation_pack.data_ptr<uint8_t>() + job.pack_offset);
      } catch (...) {
        fail();
      }
    }
    for (int64_t column = 0; column < h; column += stripe) {
      const int64_t width = std::min(stripe, h - column);
#pragma omp for schedule(dynamic, 1)
      for (size_t j = 0; j < jobs.size(); ++j) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
          auto job = jobs[j];
          auto u = kernel(job.count);
          u.run(job.count,
                width,
                inter,
                32,
                activation_pack.const_data_ptr<uint8_t>() + job.pack_offset,
                w2.const_data_ptr<uint8_t>() + job.e * w2.size(1) + u.rhs_offset(column, inter, 32),
                partial.data_ptr<float>() + job.start * stripe,
                stripe * 4,
                4,
                -inf,
                inf);
        } catch (...) {
          fail();
        }
      }
// Fold weighting into ordered reduction. -ffp-contract=off retains the
// original separate FP32 multiplication and addition (no reassociation).
#pragma omp for schedule(static)
      for (int64_t t = 0; t < tokens; ++t) {
        if (failed.load(std::memory_order_relaxed)) continue;
        for (int64_t c = 0; c < width; c += 4) {
          auto sum = vdupq_n_f32(0);
          for (int64_t r = 0; r < top; ++r) {
            auto p = positions[t * top + r];
            auto value = vmulq_f32(vld1q_f32(partial.const_data_ptr<float>() + p * stripe + c),
                                   vdupq_n_f32(expert_gates[p]));
            sum = vaddq_f32(sum, value);
          }
          vst1q_f32(output.data_ptr<float>() + t * h + column + c, sum);
        }
      }
    }
  }
  if (error) std::rethrow_exception(error);
  return output.to(input.scalar_type());
}
at::Tensor streamed_raw(const at::Tensor& input,
                        const at::Tensor& logits,
                        const at::Tensor& w1,
                        const at::Tensor& w2,
                        int64_t h,
                        int64_t inter,
                        int64_t top,
                        bool renormalize,
                        int64_t tile_n) {
  return flag_gems::arm::invoke_raw(
      [&] { return streamed(input, logits, w1, w2, h, inter, top, renormalize, tile_n); });
}

}  // namespace
TORCH_LIBRARY_FRAGMENT(flag_gems_arm_moe_prepack, m) {
  m.def(
      "streamed(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize, int tile_n) -> Tensor");
  m.def(
      "streamed.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize, int tile_n) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_moe_prepack, CPU, m) {
  m.impl("streamed", streamed);
  m.impl("streamed.raw", streamed_raw);
}
