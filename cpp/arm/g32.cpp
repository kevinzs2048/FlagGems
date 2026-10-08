// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "cpu_features.h"
#include "raw_dispatch.h"
// Local scheduling wrapper; arithmetic and packing use Arm KleidiAI 1.29.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/_dyn_quant_matmul_4bit.h>
#include <arm_neon.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/ParallelGuard.h>
#include <pthread.h>
#include "silu_mul.h"
#if defined(__linux__)
#include <sched.h>
#endif
#include <kai/kai_common.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#include <kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h>
#include <omp.h>
#include <torch/library.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

namespace {
constexpr size_t G = 32;
struct Kernel {
  size_t mr, nr, kr, sr;
  size_t (*lhs_offset)(size_t, size_t);
  size_t (*rhs_offset)(size_t, size_t, size_t);
  void (*run)(size_t, size_t, size_t, size_t, const void*, const void*, float*, size_t, size_t, float, float);
};
#define MAKE_KERNEL(suffix)                                                                    \
  Kernel {                                                                                     \
    kai_get_mr_matmul_clamp_f32_##suffix(), kai_get_nr_matmul_clamp_f32_##suffix(),            \
        kai_get_kr_matmul_clamp_f32_##suffix(), kai_get_sr_matmul_clamp_f32_##suffix(),        \
        kai_get_lhs_packed_offset_matmul_clamp_f32_##suffix,                                   \
        kai_get_rhs_packed_offset_matmul_clamp_f32_##suffix, kai_run_matmul_clamp_f32_##suffix \
  }
Kernel kernel(bool sme, bool dot) {
  if (sme) {
    flag_gems::arm::check_matrix_feature(true, dot);
    if (dot) return MAKE_KERNEL(qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot);
    return MAKE_KERNEL(qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa);
  }
  if (dot) return MAKE_KERNEL(qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod);
  return MAKE_KERNEL(qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm);
}
size_t rhs_size(int64_t n, int64_t k, bool sme) {
  if (!n) return 0;
  auto u = kernel(sme, false);
  if (sme)
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(n,
                                                                                  k,
                                                                                  u.nr,
                                                                                  u.kr,
                                                                                  u.sr,
                                                                                  G,
                                                                                  kai_dt_bf16);
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(n, k, u.nr, u.kr, u.sr, G, kai_dt_bf16);
}
at::Tensor pack(const at::Tensor& w, const at::Tensor& scales, int64_t split) {
  TORCH_CHECK(w.device().is_cpu() && w.is_contiguous() && w.scalar_type() == at::kChar &&
                  (w.dim() == 2 || w.dim() == 3),
              "INT8 [N,K] or [E,N,K] required");
  int64_t e = w.dim() == 3 ? w.size(0) : 1, n = w.size(-2), k = w.size(-1);
  TORCH_CHECK(n > 0 && k > 0 && k % G == 0 && split >= 0 && split <= n && split % 64 == 0 && n % 8 == 0,
              "Invalid G32 dimensions/split");
  TORCH_CHECK(scales.device().is_cpu() && scales.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                  scales.numel() == e * n * k / G,
              "Invalid G32 BF16 scales");
  auto offset = rhs_size(split, k, true), stride = offset + rhs_size(n - split, k, false);
  auto out = at::empty({e, (int64_t)stride}, w.options().dtype(at::kByte));
  auto nibbles = at::empty({e, n, k / 2}, w.options().dtype(at::kByte));
  auto src = w.const_data_ptr<int8_t>();
  auto dst = nibbles.data_ptr<uint8_t>();
  at::parallel_for(0, e * n * k / 2, 4096, [&](int64_t a, int64_t b) {
    for (auto i = a; i < b; ++i) {
      int lo = src[2 * i] + 8, hi = src[2 * i + 1] + 8;
      TORCH_CHECK(lo >= 0 && lo < 16 && hi >= 0 && hi < 16, "INT4 code out of range");
      dst[i] = lo | (hi << 4);
    }
  });
  kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params {1, 8, kai_dt_bf16};
  at::parallel_for(0, e, 1, [&](int64_t a, int64_t b) {
    for (auto i = a; i < b; ++i) {
      for (int side = 0; side < 2; ++side) {
        bool sme = side == 0;
        auto width = sme ? split : n - split;
        if (!width) continue;
        auto u = kernel(sme, false);
        auto col = sme ? 0 : split;
        const auto* q = dst + (i * n + col) * k / 2;
        const auto* s = scales.const_data_ptr<at::BFloat16>() + (i * n + col) * k / G;
        auto* p = out.data_ptr<uint8_t>() + i * stride + (sme ? 0 : offset);
        if (sme)
          kai_run_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(1,
                                                                 width,
                                                                 k,
                                                                 u.nr,
                                                                 u.kr,
                                                                 u.sr,
                                                                 G,
                                                                 q,
                                                                 k / 2,
                                                                 nullptr,
                                                                 s,
                                                                 k / G * 2,
                                                                 p,
                                                                 0,
                                                                 &params);
        else
          kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1,
                                                    width,
                                                    k,
                                                    u.nr,
                                                    u.kr,
                                                    u.sr,
                                                    G,
                                                    q,
                                                    k / 2,
                                                    nullptr,
                                                    s,
                                                    k / G * 2,
                                                    p,
                                                    0,
                                                    &params);
      }
    }
  });
  return out;
}
struct Tile {
  int64_t row, rows, col, cols;
};
std::vector<Tile> tiles(int64_t m, int64_t n, Kernel u, bool sme) {
  std::vector<Tile> t;
  int64_t mt = m == 1 ? 1 : (sme ? 64 : 32), nt = sme ? 256 : 128;
  mt = (mt + u.mr - 1) / u.mr * u.mr;
  nt = (nt + u.nr - 1) / u.nr * u.nr;
  for (int64_t a = 0; a < m; a += mt)
    for (int64_t b = 0; b < n; b += nt) t.push_back({a, std::min(mt, m - a), b, std::min(nt, n - b)});
  return t;
}
struct DenseG32Plan {
  const void* input;
  const uint8_t* rhs;
  uint8_t* lhs;
  float* output;
  int64_t m, n, k, split, workers, sme_workers;
  bool shared_pack, canonical_input;
  size_t offset, size_sm;
  Kernel ne, sm;
  const std::vector<Tile>* sme_tiles;
  const std::vector<Tile>* neon_tiles;
  std::exception_ptr error;
};

}  // namespace
extern "C" void flag_gems_arm_g32_compute(uint64_t descriptor) noexcept {
  auto& plan = *reinterpret_cast<DenseG32Plan*>(descriptor);
  try {
    const auto m = plan.m, n = plan.n, k = plan.k, split = plan.split;
    const auto workers = plan.workers, sme_workers = plan.sme_workers;
    const auto shared_pack = plan.shared_pack, canonical_input = plan.canonical_input;
    const auto offset = plan.offset, size_sm = plan.size_sm;
    const auto& ne = plan.ne;
    const auto& sm = plan.sm;
    const auto& st = *plan.sme_tiles;
    const auto& nt = *plan.neon_tiles;
    const auto* input = plan.input;
    const auto* rhs = plan.rhs;
    auto* lhs = plan.lhs;
    auto* output = plan.output;
    std::atomic<size_t> si {0}, ni {0};
    struct alignas(128) Claim {
      std::atomic<int> active {0};
    };
    // Three explicitly selected Apple clusters plus the unknown-CPU fallback.
    // No path indexes beyond 3; avoid initializing 28 unused cache lines.
    Claim claims[4];
    // A single Neon input row has no parallel packing work. Prepare it before
    // forking so workers do not enter a worksharing loop and packing barrier.
    const bool serial_neon_pack = m == 1 && split == 0 && !canonical_input;
    if (serial_neon_pack)
      kai_run_lhs_quant_pack_qai8dxp_f32(1,
                                         k,
                                         ne.mr,
                                         ne.kr,
                                         ne.sr,
                                         0,
                                         static_cast<const float*>(input),
                                         k * 4,
                                         lhs);
    // Small Neon calls have one M tile. Workers beyond both the packing
    // tasks and output tiles can only spin at barriers and compete for CPUs.
    // Preserve packing parallelism when it exceeds the number of GEMM tiles.
    auto team_workers = workers;
    if (split == 0 && m <= 32) {
      const int64_t packing_tasks = serial_neon_pack ? 1 : (m + ne.mr - 1) / ne.mr;
      team_workers = std::min(workers, std::max<int64_t>(nt.size(), packing_tasks));
    }
// Quantizers use identical per-row mathematics; layouts have separate packs.
// Output regions are disjoint, so SME2 and Neon GEMMs can execute together.
#pragma omp parallel num_threads(team_workers)
    {
      if (serial_neon_pack) {
        // The fork publishes the prepared LHS to every worker.
      } else if (canonical_input) {
        // Layout conversion only: share the exact Runtime A8 values and metadata
        // with SME2 and Neon, without a second activation quantization.
        int64_t step = std::max(sm.mr, ne.mr);
#pragma omp for schedule(static)
        for (int64_t block = 0; block < m; block += step) {
          for (int64_t row = block; row < std::min(m, block + step); ++row) {
            const auto* src = static_cast<const uint8_t*>(input) + row * (k + 8);
            for (int side = 0; side < 2; ++side) {
              bool sme = side == 0;
              if ((sme && !split) || (!sme && split == n)) continue;
              auto u = sme ? sm : ne;
              int64_t width = u.kr / u.sr, slot = row % u.mr;
              auto* dst = lhs + (sme ? 0 : size_sm) + u.lhs_offset(row / u.mr * u.mr, k);
              for (int64_t c = 0; c < k; c += width)
                std::memcpy(dst + (c / width * u.mr + slot) * width, src + c, width);
              std::memcpy(dst + u.mr * k + slot * 4, src + k, 4);
              std::memcpy(dst + u.mr * (k + 4) + slot * 4, src + k + 4, 4);
            }
          }
        }
      } else if (shared_pack && split > 0 && split < n) {
        // One canonical A8 quantization per row, then copy into both KAI layouts.
        // Assign whole MR groups to one thread to avoid inter-row false sharing.
        std::vector<uint8_t> canonical(k + 8);
        int64_t step = std::max(sm.mr, ne.mr);
#pragma omp for schedule(static)
        for (int64_t block = 0; block < m; block += step) {
          for (int64_t row = block; row < std::min(m, block + step); ++row) {
            kai_run_lhs_quant_pack_qai8dxp_f32(1,
                                               k,
                                               1,
                                               8,
                                               1,
                                               0,
                                               static_cast<const float*>(input) + row * k,
                                               k * 4,
                                               canonical.data());
            auto* dst_sm = lhs + sm.lhs_offset(row / sm.mr * sm.mr, k);
            auto* dst_ne = lhs + size_sm + ne.lhs_offset(row / ne.mr * ne.mr, k);
            size_t rs = row % sm.mr, rn = row % ne.mr;
            for (int64_t c = 0; c < k; c += 8) {
              std::memcpy(dst_sm + (c / 4 * sm.mr + rs) * 4, canonical.data() + c, 4);
              std::memcpy(dst_sm + ((c / 4 + 1) * sm.mr + rs) * 4, canonical.data() + c + 4, 4);
              std::memcpy(dst_ne + (c / 8 * ne.mr + rn) * 8, canonical.data() + c, 8);
            }
            std::memcpy(dst_sm + sm.mr * k + rs * 4, canonical.data() + k, 4);
            std::memcpy(dst_sm + sm.mr * (k + 4) + rs * 4, canonical.data() + k + 4, 4);
            std::memcpy(dst_ne + ne.mr * k + rn * 4, canonical.data() + k, 4);
            std::memcpy(dst_ne + ne.mr * (k + 4) + rn * 4, canonical.data() + k + 4, 4);
          }
        }
      } else {
        // The two packing regions write disjoint buffers. Skip an unused
        // region and synchronize once after both, before any GEMM reads LHS.
        if (split) {
#pragma omp for schedule(static) nowait
          for (int64_t row = 0; row < m; row += sm.mr) {
            kai_run_lhs_quant_pack_qai8dxp_f32(std::min<int64_t>(sm.mr, m - row),
                                               k,
                                               sm.mr,
                                               sm.kr,
                                               sm.sr,
                                               0,
                                               static_cast<const float*>(input) + row * k,
                                               k * 4,
                                               lhs + sm.lhs_offset(row, k));
          }
        }
        if (split < n) {
#pragma omp for schedule(static) nowait
          for (int64_t row = 0; row < m; row += ne.mr) {
            kai_run_lhs_quant_pack_qai8dxp_f32(std::min<int64_t>(ne.mr, m - row),
                                               k,
                                               ne.mr,
                                               ne.kr,
                                               ne.sr,
                                               0,
                                               static_cast<const float*>(input) + row * k,
                                               k * 4,
                                               lhs + size_sm + ne.lhs_offset(row, k));
          }
        }
#pragma omp barrier
      }
      const float inf = std::numeric_limits<float>::infinity();
      auto run = [&](bool sme, size_t i) {
        auto tile = (sme ? st : nt)[i];
        auto u = sme ? sm : ne;
        u.run(tile.rows,
              tile.cols,
              k,
              G,
              lhs + (sme ? 0 : size_sm) + u.lhs_offset(tile.row, k),
              rhs + (sme ? 0 : offset) + u.rhs_offset(tile.col, k, G),
              output + tile.row * n + (sme ? 0 : split) + tile.col,
              n * 4,
              4,
              -inf,
              inf);
      };
      if (sme_workers == -1 && split > 0 && split < n) {
        // M5 Pro mapping from the existing verified toolchain: 3 clusters of 6 CPUs.
        while (true) {
          if (si.load(std::memory_order_relaxed) < st.size()) {
            size_t cpu = 0;
#if defined(__APPLE__)
            bool known = pthread_cpu_number_np(&cpu) == 0 && cpu < 18;
#else
            bool known = false;  // Apple cluster policy is not a Linux topology.
#endif
            int cluster = known ? cpu / 6 : 3, expected = 0;
            if (claims[cluster].active.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
              auto i = si.fetch_add(1);
              if (i < st.size()) run(true, i);
              claims[cluster].active.store(0, std::memory_order_release);
              if (i < st.size()) continue;
            }
          }
          auto i = ni.fetch_add(1);
          if (i < nt.size()) {
            run(false, i);
            continue;
          }
          break;
        }
      } else if (split == 0 && m <= 32 && n % 128 == 0) {
        // One row tile and equal-width Neon tiles have uniform work. Stable
        // ownership avoids a contended claim counter and reuses each worker's
        // weight tiles across decode calls. Mixed/partial tiles keep stealing.
        const size_t tid = omp_get_thread_num(), team = omp_get_num_threads();
        const size_t chunk = nt.size() / team, extra = nt.size() % team;
        const size_t begin = tid * chunk + std::min(tid, extra);
        const size_t end = begin + chunk + (tid < extra);
        for (size_t i = begin; i < end; ++i) run(false, i);
      } else {
        auto drain = [&](bool sme) {
          auto& queue = sme ? si : ni;
          auto& work = sme ? st : nt;
          for (size_t i = queue.fetch_add(1); i < work.size(); i = queue.fetch_add(1)) run(sme, i);
        };
        if (split > 0 && split < n) {
          // OpenMP can grant fewer workers (thread limit or a nested region).
          // Keep both output partitions owned, including a serialized team.
          const int64_t actual_team = omp_get_num_threads();
          if (actual_team == 1) {
            drain(true);
            drain(false);
          } else {
            const auto active_sme_workers = std::min(sme_workers, actual_team - 1);
            drain(omp_get_thread_num() < active_sme_workers);
          }
        } else {
          drain(split == n);
        }
      }
    }
  } catch (...) {
    plan.error = std::current_exception();
  }
}
namespace {

__attribute__((noinline)) at::Tensor linear_impl(const at::Tensor& x,
                                                 const at::Tensor& rhs,
                                                 int64_t n,
                                                 int64_t split,
                                                 int64_t workers,
                                                 int64_t sme_workers,
                                                 bool shared_pack,
                                                 bool canonical_input = false,
                                                 bool raw = false) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == (canonical_input ? at::kByte : at::kFloat) &&
                  x.dim() == 2 && x.is_contiguous(),
              "Invalid contiguous input container");
  int64_t m = x.size(0), k = x.size(1) - (canonical_input ? 8 : 0);
  TORCH_CHECK(m > 0 && k > 0 && k % G == 0 && n > 0 && split >= 0 && split <= n && split % 64 == 0 &&
                  workers > 0 && workers <= 64,
              "Invalid linear configuration");
  TORCH_CHECK(split == 0 || split == n || sme_workers == -1 || (sme_workers > 0 && sme_workers < workers),
              "Hybrid mode needs workers for both kernels");
  if (sme_workers == -1 && split > 0 && split < n) {
#if defined(__APPLE__)
    TORCH_CHECK(std::getenv("FLAGGEMS_ARM_APPLE_18CPU_CLUSTER_POLICY") &&
                    std::strcmp(std::getenv("FLAGGEMS_ARM_APPLE_18CPU_CLUSTER_POLICY"), "1") == 0,
                "Implicit 18-CPU cluster scheduling requires explicit platform policy");
#else
    TORCH_CHECK(false, "Linux hybrid scheduling requires an explicit SME worker count");
#endif
  }
  size_t offset = rhs_size(split, k, true);
  TORCH_CHECK(rhs.device().is_cpu() && rhs.is_contiguous() && rhs.scalar_type() == at::kByte &&
                  rhs.numel() == offset + rhs_size(n - split, k, false),
              "Packed RHS shape mismatch");
  // SME tile getters execute streaming instructions. Do not query them on
  // Neon-only hosts, even when the SME output region has zero width.
  if (split < n) flag_gems::arm::check_matrix_feature(false, m == 1);
  Kernel ne = kernel(false, m == 1), sm = split ? kernel(true, m == 1) : ne;
  if (shared_pack || canonical_input) {
    TORCH_CHECK((!split || sm.kr / sm.sr == 4) && (split == n || ne.kr / ne.sr == 8),
                "Unsupported shared LHS lane layout");
    if (split && split < n)
      TORCH_CHECK(std::max(sm.mr, ne.mr) % std::min(sm.mr, ne.mr) == 0, "Unsupported mixed LHS row layout");
  }
  auto size_sm = split ? kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, sm.mr, sm.kr, sm.sr) : 0;
  auto size_ne =
      split < n ? kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, ne.mr, ne.kr, ne.sr) : 0;
  auto lhs = at::empty({(int64_t)(size_sm + size_ne)}, x.options().dtype(at::kByte));
  auto output = at::empty({m, n}, x.options().dtype(at::kFloat));
  auto st = tiles(m, split, sm, true), nt = tiles(m, n - split, ne, false);
  DenseG32Plan plan {x.const_data_ptr(),
                     rhs.const_data_ptr<uint8_t>(),
                     lhs.data_ptr<uint8_t>(),
                     output.data_ptr<float>(),
                     m,
                     n,
                     k,
                     split,
                     workers,
                     sme_workers,
                     shared_pack,
                     canonical_input,
                     offset,
                     size_sm,
                     ne,
                     sm,
                     &st,
                     &nt,
                     {}};
  if (raw)
    flag_gems::arm::invoke_dense_compute(plan, 0);
  else
    flag_gems_arm_g32_compute(reinterpret_cast<uint64_t>(&plan));
  if (plan.error) std::rethrow_exception(plan.error);
  return output;
}
at::Tensor linear(const at::Tensor& x,
                  const at::Tensor& rhs,
                  int64_t n,
                  int64_t split,
                  int64_t workers,
                  int64_t sme_workers) {
  return linear_impl(x, rhs, n, split, workers, sme_workers, false);
}
at::Tensor linear_shared(const at::Tensor& x,
                         const at::Tensor& rhs,
                         int64_t n,
                         int64_t split,
                         int64_t workers,
                         int64_t sme_workers) {
  return linear_impl(x, rhs, n, split, workers, sme_workers, true);
}
at::Tensor linear_a8(const at::Tensor& x,
                     const at::Tensor& rhs,
                     int64_t n,
                     int64_t split,
                     int64_t workers,
                     int64_t sme_workers) {
  return linear_impl(x, rhs, n, split, workers, sme_workers, false, true);
}
// Decode keeps the original ATen/KleidiAI packing and matmuls. Avoid gathering
// one repeated row and scatter_add; reduce in the original ascending-expert order.
at::Tensor moe_decode_impl(const at::Tensor& x,
                           const at::Tensor& ids,
                           const at::Tensor& gates,
                           const at::Tensor& w1,
                           const at::Tensor& w2,
                           int64_t h,
                           int64_t inter,
                           int64_t group,
                           bool vector_reduce,
                           bool fused_activation = false) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat && x.is_contiguous() &&
                  x.sizes() == at::IntArrayRef({1, h}),
              "Expected FP32 [1,H]");
  TORCH_CHECK(h > 0 && inter > 0 && group == 32 && h % group == 0 && inter % group == 0,
              "Expected positive G32 dimensions");
  TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == at::kLong && ids.is_contiguous() &&
                  ids.dim() == 2 && ids.size(0) == 1 && gates.device().is_cpu() &&
                  gates.scalar_type() == at::kFloat && gates.is_contiguous() && gates.sizes() == ids.sizes(),
              "Invalid routing tensors");
  TORCH_CHECK(w1.device().is_cpu() && w2.device().is_cpu() && w1.is_contiguous() && w2.is_contiguous() &&
                  w1.dim() == 2 && w2.dim() == 2 && w1.size(0) == w2.size(0),
              "Invalid packed expert tensors");
  auto top = ids.numel();
  auto id = ids.const_data_ptr<int64_t>();
  auto g = gates.const_data_ptr<float>();
  std::vector<int64_t> order(top);
  for (int64_t i = 0; i < top; ++i) {
    order[i] = i;
    TORCH_CHECK(id[i] >= 0 && id[i] < w1.size(0), "Expert out of range");
  }
  TORCH_CHECK(top > 0 && top <= w1.size(0), "Invalid experts per token");
  std::sort(order.begin(), order.end(), [&](auto a, auto b) { return id[a] < id[b]; });
  for (int64_t i = 1; i < top; ++i)
    TORCH_CHECK(id[order[i - 1]] != id[order[i]], "Duplicate expert IDs are not supported");
  auto partial = at::empty({top, h}, x.options());
  auto out = at::empty({1, h}, x.options());
  auto expert = [&](int64_t i) {
    auto e = id[order[i]];
    auto gu = at::_ops::_dyn_quant_matmul_4bit::call(x, w1[e], group, h, 2 * inter);
    auto act =
        fused_activation ? silu_mul(gu) : at::silu(gu.narrow(1, 0, inter)).mul_(gu.narrow(1, inter, inter));
    auto y = at::_ops::_dyn_quant_matmul_4bit::call(act, w2[e], group, inter, h);
    const float* src = y.const_data_ptr<float>();
    float* dst = partial.data_ptr<float>() + i * h;
    float gate = g[order[i]];
    for (int64_t c = 0; c < h; ++c) dst[c] = src[c] * gate;
  };
  const char* threads_env = std::getenv("FLAGGEMS_ARM_MOE_DECODE_THREADS");
  int decode_threads = 0;
  if (threads_env) {
    char* end = nullptr;
    long configured = std::strtol(threads_env, &end, 10);
    TORCH_CHECK(end != threads_env && *end == '\0' && configured >= 0 && configured <= 64,
                "FLAGGEMS_ARM_MOE_DECODE_THREADS must be in [0,64]");
    decode_threads = std::min<int64_t>(configured, top);
  }
  if (decode_threads) {
    // Cyclic expert assignment uses every worker when Top-K is not divisible
    // by the decode team. omp-for also covers all experts under thread limits.
    std::exception_ptr failure;
    std::atomic_flag failed = ATOMIC_FLAG_INIT;
#pragma omp parallel num_threads(decode_threads)
    {
      c10::InferenceMode inference_guard;
      at::internal::ThreadIdGuard thread_guard(omp_get_thread_num());
      c10::ParallelGuard parallel_guard(true);
#pragma omp for schedule(static, 1)
      for (int64_t i = 0; i < top; ++i) {
        try {
          expert(i);
        } catch (...) {
          if (!failed.test_and_set()) failure = std::current_exception();
        }
      }
    }
    if (failure) std::rethrow_exception(failure);
  } else {
    at::parallel_for(0, top, 1, [&](int64_t begin, int64_t end) {
      c10::InferenceMode guard;
      for (int64_t i = begin; i < end; ++i) expert(i);
    });
  }
  // Vectorize channels, retain expert summation order and FP32 rounding.
  const float* p = partial.const_data_ptr<float>();
  float* o = out.data_ptr<float>();
  if (vector_reduce) {
    // Vectorize independent channels, never reassociate the expert reduction.
    // h is G32-aligned, so there is no vector tail.
    for (int64_t c = 0; c < h; c += 4) {
      float32x4_t sum = vdupq_n_f32(0);
      for (int64_t i = 0; i < top; ++i) sum = vaddq_f32(sum, vld1q_f32(p + i * h + c));
      vst1q_f32(o + c, sum);
    }
  } else {
    for (int64_t c = 0; c < h; ++c) {
      float sum = 0;
      for (int64_t i = 0; i < top; ++i) sum += p[i * h + c];
      o[c] = sum;
    }
  }
  return out;
}
at::Tensor moe_decode(const at::Tensor& x,
                      const at::Tensor& ids,
                      const at::Tensor& gates,
                      const at::Tensor& w1,
                      const at::Tensor& w2,
                      int64_t h,
                      int64_t inter,
                      int64_t group) {
  return moe_decode_impl(x, ids, gates, w1, w2, h, inter, group, false);
}
at::Tensor moe_decode_routed_impl(const at::Tensor& x,
                                  const at::Tensor& logits,
                                  const at::Tensor& w1,
                                  const at::Tensor& w2,
                                  int64_t h,
                                  int64_t inter,
                                  int64_t top,
                                  bool renormalize,
                                  bool fused_activation) {
  TORCH_CHECK(x.device().is_cpu() && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16),
              "Expected CPU FP32/BF16 input");
  TORCH_CHECK(
      logits.device().is_cpu() && logits.dim() == 2 && logits.size(0) == 1 && logits.size(1) == w1.size(0),
      "Invalid router logits");
  TORCH_CHECK(top > 0 && top <= logits.size(1), "Invalid top-k");
  // Use exactly the ATen routing operations of vLLM select_experts. Keep the
  // original logits dtype and unsorted top-k order, including ties.
  auto selected = at::topk(logits, top, -1, true, false);
  auto values = std::get<0>(selected), ids = std::get<1>(selected);
  auto gates = renormalize ? at::softmax(values, -1) : at::exp(values - at::logsumexp(logits, {-1}, true));
  auto result = moe_decode_impl(x.to(at::kFloat).contiguous(),
                                ids.contiguous(),
                                gates.to(at::kFloat).contiguous(),
                                w1,
                                w2,
                                h,
                                inter,
                                32,
                                true,
                                fused_activation);
  return result.to(x.scalar_type());
}
at::Tensor moe_decode_routed(const at::Tensor& x,
                             const at::Tensor& logits,
                             const at::Tensor& w1,
                             const at::Tensor& w2,
                             int64_t h,
                             int64_t inter,
                             int64_t top,
                             bool renormalize) {
  return moe_decode_routed_impl(x, logits, w1, w2, h, inter, top, renormalize, false);
}
at::Tensor moe_decode_fused(const at::Tensor& x,
                            const at::Tensor& logits,
                            const at::Tensor& w1,
                            const at::Tensor& w2,
                            int64_t h,
                            int64_t inter,
                            int64_t top,
                            bool renormalize) {
  return moe_decode_routed_impl(x, logits, w1, w2, h, inter, top, renormalize, true);
}
at::Tensor linear_raw(const at::Tensor& x,
                      const at::Tensor& rhs,
                      int64_t n,
                      int64_t split,
                      int64_t workers,
                      int64_t sme_workers) {
  return linear_impl(x, rhs, n, split, workers, sme_workers, false, false, true);
}
at::Tensor linear_shared_raw(const at::Tensor& x,
                             const at::Tensor& rhs,
                             int64_t n,
                             int64_t split,
                             int64_t workers,
                             int64_t sme_workers) {
  return linear_impl(x, rhs, n, split, workers, sme_workers, true, false, true);
}
at::Tensor linear_a8_raw(const at::Tensor& x,
                         const at::Tensor& rhs,
                         int64_t n,
                         int64_t split,
                         int64_t workers,
                         int64_t sme_workers) {
  return linear_impl(x, rhs, n, split, workers, sme_workers, false, true, true);
}
at::Tensor moe_decode_raw(const at::Tensor& x,
                          const at::Tensor& ids,
                          const at::Tensor& gates,
                          const at::Tensor& w1,
                          const at::Tensor& w2,
                          int64_t h,
                          int64_t inter,
                          int64_t group) {
  return flag_gems::arm::invoke_raw([&] { return moe_decode(x, ids, gates, w1, w2, h, inter, group); });
}
at::Tensor moe_decode_routed_raw(const at::Tensor& x,
                                 const at::Tensor& logits,
                                 const at::Tensor& w1,
                                 const at::Tensor& w2,
                                 int64_t h,
                                 int64_t inter,
                                 int64_t top,
                                 bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_decode_routed(x, logits, w1, w2, h, inter, top, renormalize); });
}
at::Tensor moe_decode_fused_raw(const at::Tensor& x,
                                const at::Tensor& logits,
                                const at::Tensor& w1,
                                const at::Tensor& w2,
                                int64_t h,
                                int64_t inter,
                                int64_t top,
                                bool renormalize) {
  return flag_gems::arm::invoke_raw(
      [&] { return moe_decode_fused(x, logits, w1, w2, h, inter, top, renormalize); });
}

}  // namespace
TORCH_LIBRARY(flag_gems_arm_g32, m) {
  m.def("set_threads(int threads) -> ()");
  m.def("pack(Tensor weight, Tensor scales, int split) -> Tensor");
  m.def("linear(Tensor input, Tensor rhs, int n, int split, int workers, int sme_workers) -> Tensor");
  m.def("linear.raw(Tensor input, Tensor rhs, int n, int split, int workers, int sme_workers) -> Tensor");
  m.def("linear_shared(Tensor input, Tensor rhs, int n, int split, int workers, int sme_workers) -> Tensor");
  m.def(
      "linear_shared.raw(Tensor input, Tensor rhs, int n, int split, int workers, int sme_workers) -> "
      "Tensor");
  m.def("linear_a8(Tensor input, Tensor rhs, int n, int split, int workers, int sme_workers) -> Tensor");
  m.def("linear_a8.raw(Tensor input, Tensor rhs, int n, int split, int workers, int sme_workers) -> Tensor");
  m.def(
      "moe_decode(Tensor x, Tensor ids, Tensor gates, Tensor w1, Tensor w2, int h, int inter, int group) -> "
      "Tensor");
  m.def(
      "moe_decode.raw(Tensor x, Tensor ids, Tensor gates, Tensor w1, Tensor w2, int h, int inter, int group) "
      "-> Tensor");
  m.def(
      "moe_decode_routed(Tensor x, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_decode_routed.raw(Tensor x, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_decode_fused(Tensor x, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
  m.def(
      "moe_decode_fused.raw(Tensor x, Tensor logits, Tensor w1, Tensor w2, int h, int inter, int top, bool "
      "renormalize) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_g32, CatchAll, m) {
  m.impl("set_threads", [](int64_t n) {
    TORCH_CHECK(n > 0 && n <= 64, "Invalid CPU thread count");
    if (at::get_num_threads() != n) at::set_num_threads(n);
  });
}
TORCH_LIBRARY_IMPL(flag_gems_arm_g32, CPU, m) {
  m.impl("pack", pack);
  m.impl("linear", linear);
  m.impl("linear.raw", linear_raw);
  m.impl("linear_shared", linear_shared);
  m.impl("linear_shared.raw", linear_shared_raw);
  m.impl("linear_a8", linear_a8);
  m.impl("linear_a8.raw", linear_a8_raw);
  m.impl("moe_decode", moe_decode);
  m.impl("moe_decode.raw", moe_decode_raw);
  m.impl("moe_decode_routed", moe_decode_routed);
  m.impl("moe_decode_routed.raw", moe_decode_routed_raw);
  m.impl("moe_decode_fused", moe_decode_fused);
  m.impl("moe_decode_fused.raw", moe_decode_fused_raw);
}
