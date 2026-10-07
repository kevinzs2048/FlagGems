// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "raw_dispatch.h"
// Preserve the installed ATen expert arithmetic; replace sparse scatter with
// a token-parallel, ordered Neon reduction of the selected expert outputs.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/_dyn_quant_matmul_4bit.h>
#include <arm_neon.h>
#include <c10/core/InferenceMode.h>
#include <omp.h>
#include <torch/library.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <vector>
#include "silu_mul.h"

namespace {
at::Tensor moe_prefill_impl(const at::Tensor& input,
                            const at::Tensor& logits,
                            const at::Tensor& w1,
                            const at::Tensor& w2,
                            int64_t h,
                            int64_t inter,
                            int64_t top,
                            bool renormalize,
                            bool local_gather,
                            bool fused_activation,
                            int dynamic_mode = 0,
                            int64_t tile_rows = 0) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2 && input.size(1) == h &&
                  (input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16),
              "Expected CPU FP32/BF16 [T,H]");
  TORCH_CHECK(h > 0 && inter > 0 && h % 32 == 0 && inter % 32 == 0, "Expected positive G32 dimensions");
  if (dynamic_mode == 3)
    TORCH_CHECK(tile_rows >= 4 && tile_rows <= 128 && (tile_rows & (tile_rows - 1)) == 0,
                "Tile rows must be a power of two in [4,128]");
  TORCH_CHECK(w1.device().is_cpu() && w2.device().is_cpu() && w1.is_contiguous() && w2.is_contiguous() &&
                  w1.dim() == 2 && w2.dim() == 2 && w1.size(0) == w2.size(0),
              "Invalid packed expert tensors");
  const int64_t tokens = input.size(0), experts = w1.size(0), routes = tokens * top;
  TORCH_CHECK(tokens > 0 && top > 0 && top <= experts && logits.device().is_cpu() &&
                  logits.sizes() == at::IntArrayRef({tokens, experts}),
              "Invalid routing dimensions");
  auto selected = at::topk(logits, top, -1, true, false);
  auto values = std::get<0>(selected), ids = std::get<1>(selected).contiguous();
  auto gates = (renormalize ? at::softmax(values, -1) : at::exp(values - at::logsumexp(logits, {-1}, true)))
                   .to(at::kFloat)
                   .contiguous();
  auto x = input.to(at::kFloat).contiguous();
  const auto* id = ids.const_data_ptr<int64_t>();
  const auto* g = gates.const_data_ptr<float>();
  std::vector<int64_t> offsets(experts + 1, 0), cursor(experts, 0), positions(routes);
  for (int64_t i = 0; i < routes; ++i) ++offsets[id[i] + 1];
  for (int64_t e = 0; e < experts; ++e) offsets[e + 1] += offsets[e];
  auto token_ids = at::empty({routes}, ids.options());
  auto expert_gates = at::empty({routes}, gates.options());
  auto* tid = token_ids.data_ptr<int64_t>();
  auto* eg = expert_gates.data_ptr<float>();
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t r = 0; r < top; ++r) {
      auto idx = t * top + r, e = id[idx], p = offsets[e] + cursor[e]++;
      tid[p] = t;
      eg[p] = g[idx];
      positions[idx] = p;
    }
    // Positions are in ascending expert order, matching original index_add.
    std::sort(positions.begin() + t * top, positions.begin() + (t + 1) * top);
  }
  auto gathered = local_gather ? at::Tensor() : x.index_select(0, token_ids);
  auto partial = at::empty({routes, h}, x.options());
  const int team = (at::in_parallel_region() || omp_in_parallel()) ? 1 : at::get_num_threads();
  int64_t active_experts = 0;
  if (dynamic_mode == 2)
    for (int64_t e = 0; e < experts; ++e) active_experts += offsets[e + 1] > offsets[e];
  int64_t tile_jobs = 0;
  if (dynamic_mode == 4) {
    int64_t jobs64 = 0;
    for (int64_t e = 0; e < experts; ++e) jobs64 += (offsets[e + 1] - offsets[e] + 63) / 64;
    tile_rows = jobs64 > 4 * team ? 64 : 16;
  }
  if (dynamic_mode >= 3)
    for (int64_t e = 0; e < experts; ++e)
      tile_jobs += (offsets[e + 1] - offsets[e] + tile_rows - 1) / tile_rows;
  // Few large jobs need the original balanced route chunks. Dynamic scheduling
  // helps when there are enough independent expert GEMMs to refill workers.
  bool dynamic_jobs = dynamic_mode == 1 || (dynamic_mode == 2 && active_experts >= 4 * team) ||
                      (dynamic_mode == 3 && tile_jobs >= 2 * team);
  dynamic_jobs = dynamic_jobs || (dynamic_mode == 4 && tile_jobs > 4 * team);
  auto run_expert = [&](int64_t e, int64_t start, int64_t count) {
    at::Tensor expert_input;
    if (local_gather) {
      // Gather just this GEMM's rows into a thread-local allocation, consume
      // them immediately, then release. Avoid the full T*TopK*H gather.
      expert_input = at::empty({count, h}, x.options());
      const float* src = x.const_data_ptr<float>();
      auto* dst = expert_input.data_ptr<float>();
      for (int64_t row = 0; row < count; ++row)
        std::memcpy(dst + row * h, src + tid[start + row] * h, h * sizeof(float));
    } else {
      expert_input = gathered.narrow(0, start, count);
    }
    auto gu = at::_ops::_dyn_quant_matmul_4bit::call(expert_input, w1[e], 32, h, 2 * inter);
    auto act =
        fused_activation ? silu_mul(gu) : at::silu(gu.narrow(1, 0, inter)).mul_(gu.narrow(1, inter, inter));
    auto y = at::_ops::_dyn_quant_matmul_4bit::call(act, w2[e], 32, inter, h);
    const float* src = y.const_data_ptr<float>();
    for (int64_t row = 0; row < count; ++row) {
      float* dst = partial.data_ptr<float>() + (start + row) * h;
      auto scale = vdupq_n_f32(eg[start + row]);
      for (int64_t c = 0; c < h; c += 4) vst1q_f32(dst + c, vmulq_f32(vld1q_f32(src + row * h + c), scale));
    }
  };
  // Static/adaptive preserve original batch shapes; tiled modes change only
  // row blocking, with a separate bitwise GEMM/whole-expert validation gate.
  if (!dynamic_jobs) {
    at::parallel_for(0, routes, 0, [&](int64_t begin, int64_t end) {
      c10::InferenceMode guard;
      auto first = std::upper_bound(offsets.begin(), offsets.end(), begin) - offsets.begin() - 1;
      for (int64_t e = first; e < experts && offsets[e] < end; ++e) {
        auto start = std::max(offsets[e], begin), finish = std::min(offsets[e + 1], end),
             count = finish - start;
        if (count > 0) run_expert(e, start, count);
      }
    });
  } else {
    struct Job {
      int64_t expert, start, count;
    };
    std::vector<Job> jobs;
    std::atomic<size_t> next {0};
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
          int64_t actual_team = omp_get_num_threads(), chunk = (routes + actual_team - 1) / actual_team;
          if (dynamic_mode >= 3) {
            jobs.reserve(tile_jobs);
            for (int64_t e = 0; e < experts; ++e)
              for (auto start = offsets[e]; start < offsets[e + 1]; start += tile_rows)
                jobs.push_back({e, start, std::min(tile_rows, offsets[e + 1] - start)});
          } else
            for (int64_t begin = 0; begin < routes; begin += chunk) {
              auto end = std::min(routes, begin + chunk);
              auto first = std::upper_bound(offsets.begin(), offsets.end(), begin) - offsets.begin() - 1;
              for (int64_t e = first; e < experts && offsets[e] < end; ++e) {
                auto start = std::max(offsets[e], begin), finish = std::min(offsets[e + 1], end);
                if (finish > start) jobs.push_back({e, start, finish - start});
              }
            }
          if (dynamic_mode >= 2)
            std::stable_sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
              return a.count > b.count;
            });
        } catch (...) {
          failed.test_and_set();
          error = std::current_exception();
        }
      }
      // Faster workers claim more jobs; disjoint output ranges and reduction
      // order remain unchanged. ATen's nested parallelism stays disabled.
      for (auto j = next.fetch_add(1); j < jobs.size(); j = next.fetch_add(1)) {
        try {
          auto job = jobs[j];
          run_expert(job.expert, job.start, job.count);
        } catch (...) {
          if (!failed.test_and_set()) error = std::current_exception();
        }
      }
    }
    if (error) std::rethrow_exception(error);
  }
  auto output = at::empty({tokens, h}, x.options());
  const float* p = partial.const_data_ptr<float>();
  auto* out = output.data_ptr<float>();
  at::parallel_for(0, tokens, 1, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      for (int64_t c = 0; c < h; c += 4) {
        auto sum = vdupq_n_f32(0);
        for (int64_t r = 0; r < top; ++r) sum = vaddq_f32(sum, vld1q_f32(p + positions[t * top + r] * h + c));
        vst1q_f32(out + t * h + c, sum);
      }
    }
  });
  return output.to(input.scalar_type());
}
at::Tensor moe_prefill_routed(const at::Tensor& x,
                              const at::Tensor& logits,
                              const at::Tensor& w1,
                              const at::Tensor& w2,
                              int64_t h,
                              int64_t inter,
                              int64_t top,
                              bool renormalize) {
  return moe_prefill_impl(x, logits, w1, w2, h, inter, top, renormalize, false, false);
}
at::Tensor moe_prefill_local(const at::Tensor& x,
                             const at::Tensor& logits,
                             const at::Tensor& w1,
                             const at::Tensor& w2,
                             int64_t h,
                             int64_t inter,
                             int64_t top,
                             bool renormalize) {
  return moe_prefill_impl(x, logits, w1, w2, h, inter, top, renormalize, true, false);
}
at::Tensor moe_prefill_fused(const at::Tensor& x,
                             const at::Tensor& logits,
                             const at::Tensor& w1,
                             const at::Tensor& w2,
                             int64_t h,
                             int64_t inter,
                             int64_t top,
                             bool renormalize) {
  return moe_prefill_impl(x, logits, w1, w2, h, inter, top, renormalize, false, true);
}
at::Tensor moe_prefill_dynamic(const at::Tensor& x,
                               const at::Tensor& logits,
                               const at::Tensor& w1,
                               const at::Tensor& w2,
                               int64_t h,
                               int64_t inter,
                               int64_t top,
                               bool renormalize) {
  return moe_prefill_impl(x, logits, w1, w2, h, inter, top, renormalize, false, false, 1);
}
at::Tensor moe_prefill_adaptive(const at::Tensor& x,
                                const at::Tensor& logits,
                                const at::Tensor& w1,
                                const at::Tensor& w2,
                                int64_t h,
                                int64_t inter,
                                int64_t top,
                                bool renormalize) {
  return moe_prefill_impl(x, logits, w1, w2, h, inter, top, renormalize, false, false, 2);
}
at::Tensor moe_prefill_tiled(const at::Tensor& x,
                             const at::Tensor& logits,
                             const at::Tensor& w1,
                             const at::Tensor& w2,
                             int64_t h,
                             int64_t inter,
                             int64_t top,
                             bool renormalize,
                             int64_t tile_rows) {
  return moe_prefill_impl(x, logits, w1, w2, h, inter, top, renormalize, false, false, 3, tile_rows);
}
at::Tensor moe_prefill_balanced(const at::Tensor& x,
                                const at::Tensor& logits,
                                const at::Tensor& w1,
                                const at::Tensor& w2,
                                int64_t h,
                                int64_t inter,
                                int64_t top,
                                bool renormalize) {
  return moe_prefill_impl(x, logits, w1, w2, h, inter, top, renormalize, false, false, 4);
}
at::Tensor moe_prefill_routed_raw(const at::Tensor& x,
                                  const at::Tensor& logits,
                                  const at::Tensor& w1,
                                  const at::Tensor& w2,
                                  int64_t h,
                                  int64_t inter,
                                  int64_t top,
                                  bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_prefill_routed(x, logits, w1, w2, h, inter, top, renormalize); });
}
at::Tensor moe_prefill_local_raw(const at::Tensor& x,
                                 const at::Tensor& logits,
                                 const at::Tensor& w1,
                                 const at::Tensor& w2,
                                 int64_t h,
                                 int64_t inter,
                                 int64_t top,
                                 bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_prefill_local(x, logits, w1, w2, h, inter, top, renormalize); });
}
at::Tensor moe_prefill_fused_raw(const at::Tensor& x,
                                 const at::Tensor& logits,
                                 const at::Tensor& w1,
                                 const at::Tensor& w2,
                                 int64_t h,
                                 int64_t inter,
                                 int64_t top,
                                 bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_prefill_fused(x, logits, w1, w2, h, inter, top, renormalize); });
}
at::Tensor moe_prefill_dynamic_raw(const at::Tensor& x,
                                   const at::Tensor& logits,
                                   const at::Tensor& w1,
                                   const at::Tensor& w2,
                                   int64_t h,
                                   int64_t inter,
                                   int64_t top,
                                   bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_prefill_dynamic(x, logits, w1, w2, h, inter, top, renormalize); });
}
at::Tensor moe_prefill_adaptive_raw(const at::Tensor& x,
                                    const at::Tensor& logits,
                                    const at::Tensor& w1,
                                    const at::Tensor& w2,
                                    int64_t h,
                                    int64_t inter,
                                    int64_t top,
                                    bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_prefill_adaptive(x, logits, w1, w2, h, inter, top, renormalize); });
}
at::Tensor moe_prefill_tiled_raw(const at::Tensor& x,
                                 const at::Tensor& logits,
                                 const at::Tensor& w1,
                                 const at::Tensor& w2,
                                 int64_t h,
                                 int64_t inter,
                                 int64_t top,
                                 bool renormalize,
                                 int64_t tile_rows) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_prefill_tiled(x, logits, w1, w2, h, inter, top, renormalize, tile_rows); });
}
at::Tensor moe_prefill_balanced_raw(const at::Tensor& x,
                                    const at::Tensor& logits,
                                    const at::Tensor& w1,
                                    const at::Tensor& w2,
                                    int64_t h,
                                    int64_t inter,
                                    int64_t top,
                                    bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_prefill_balanced(x, logits, w1, w2, h, inter, top, renormalize); });
}

}  // namespace
TORCH_LIBRARY_FRAGMENT(flag_gems_arm_g32, m) {
  m.def(
      "moe_prefill_routed(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_prefill_routed.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_local(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_prefill_local.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_fused(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_prefill_fused.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_dynamic(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_dynamic.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_adaptive(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_adaptive.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int "
      "top, bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_tiled(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize, int tile_rows) -> Tensor");
  m.def(
      "moe_prefill_tiled.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize, int tile_rows) -> Tensor");
  m.def(
      "moe_prefill_balanced(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, "
      "bool renormalize) -> Tensor");
  m.def(
      "moe_prefill_balanced.raw(Tensor input, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int "
      "top, bool renormalize) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_g32, CPU, m) {
  m.impl("moe_prefill_routed", moe_prefill_routed);
  m.impl("moe_prefill_routed.raw", moe_prefill_routed_raw);
  m.impl("moe_prefill_local", moe_prefill_local);
  m.impl("moe_prefill_local.raw", moe_prefill_local_raw);
  m.impl("moe_prefill_fused", moe_prefill_fused);
  m.impl("moe_prefill_fused.raw", moe_prefill_fused_raw);
  m.impl("moe_prefill_dynamic", moe_prefill_dynamic);
  m.impl("moe_prefill_dynamic.raw", moe_prefill_dynamic_raw);
  m.impl("moe_prefill_adaptive", moe_prefill_adaptive);
  m.impl("moe_prefill_adaptive.raw", moe_prefill_adaptive_raw);
  m.impl("moe_prefill_tiled", moe_prefill_tiled);
  m.impl("moe_prefill_tiled.raw", moe_prefill_tiled_raw);
  m.impl("moe_prefill_balanced", moe_prefill_balanced);
  m.impl("moe_prefill_balanced.raw", moe_prefill_balanced_raw);
}
