#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <omp.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "triton_jit/triton_jit_function.h"
#include "kernel_sources.h"

namespace {

using triton_jit::TritonJITFunction;

int32_t checked_i32(int64_t value, const char* name) {
  TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() &&
                  value <= std::numeric_limits<int32_t>::max(),
              name, " does not fit the Triton CPU i32 ABI");
  return static_cast<int32_t>(value);
}

std::vector<int64_t> output_shape(const at::Tensor& input, int64_t n) {
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  TORCH_CHECK(!shape.empty(), "W8 input must have at least one dimension");
  shape.back() = n;
  return shape;
}

void validate(const at::Tensor& input,
              const at::Tensor& decode_rhs,
              const at::Tensor& prefill_rhs,
              const at::Tensor& scale,
              int64_t n,
              int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && decode_rhs.device().is_cpu() &&
                  prefill_rhs.device().is_cpu() && scale.device().is_cpu(),
              "libtriton_jit W8 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
              "libtriton_jit W8 requires contiguous BF16 input");
  TORCH_CHECK(decode_rhs.scalar_type() == at::kChar &&
                  decode_rhs.is_contiguous(),
              "libtriton_jit W8 decode RHS must be contiguous INT8");
  TORCH_CHECK(prefill_rhs.scalar_type() == at::kChar &&
                  prefill_rhs.is_contiguous(),
              "libtriton_jit W8 prefill RHS must be contiguous INT8");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.is_contiguous(),
              "libtriton_jit W8 scales must be contiguous FP32");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid W8 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 64 == 0 && k % 32 == 0,
              "W8 dimensions require N%64=0 and K%32=0");
  TORCH_CHECK(decode_rhs.numel() == n * k,
              "invalid packed W8 decode byte count");
  TORCH_CHECK(prefill_rhs.numel() == (n / 4) * (4 * k + 16),
              "invalid packed W8 prefill byte count");
  TORCH_CHECK(scale.numel() == n, "invalid W8 scale count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

void validate_kai(const at::Tensor& input,
                  const at::Tensor& rhs,
                  int64_t n,
                  int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "libtriton_jit exact-KAI W8 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
              "exact-KAI W8 requires contiguous BF16 input");
  TORCH_CHECK(rhs.scalar_type() == at::kChar && rhs.is_contiguous(),
              "exact-KAI W8 RHS must be a contiguous INT8 blob");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid exact-KAI W8 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % 32 == 0,
              "exact-KAI W8 requires N%4=0 and K%32=0");
  TORCH_CHECK(rhs.numel() == (n / 4) * (4 * k + 48),
              "invalid qsi8cxp W8 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

int32_t kai_decode_partitions(int64_t k, int64_t n) {
  if (k * n < 2 * 1024 * 1024) {
    return 1;
  }
  const int32_t threads = std::max(1, at::get_num_threads());
  if (const char* configured =
          std::getenv("FLAGGEMS_W8_DECODE_PARTITIONS")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' && parsed > 0,
                "FLAGGEMS_W8_DECODE_PARTITIONS must be a positive integer");
    return std::min<int32_t>(
        std::min<int32_t>(threads, checked_i32(parsed, "W8 partitions")),
        checked_i32(n / 4, "N/4"));
  }
  return std::min<int32_t>(
      threads, std::max<int32_t>(1, checked_i32(n / 64, "N/64")));
}

bool use_kai_stealing_decode() {
  const char* configured = std::getenv("FLAGGEMS_W8_STEALING_DECODE");
  if (configured == nullptr) {
    return true;
  }
  return std::strcmp(configured, "0") != 0 &&
      std::strcmp(configured, "false") != 0 &&
      std::strcmp(configured, "off") != 0;
}

int64_t kai_stealing_min_work() {
  const char* configured = std::getenv("FLAGGEMS_W8_STEALING_MIN_WORK");
  if (configured == nullptr) {
    return 64 * 1024 * 1024;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' && parsed >= 0,
              "FLAGGEMS_W8_STEALING_MIN_WORK must be non-negative");
  return static_cast<int64_t>(parsed);
}

// Kernel source paths are configured before the operator library is used and
// remain process-immutable.  Retaining the registry handles avoids formatting
// a key and taking libtriton_jit's global function-registry mutex for every
// linear; the registry owns these entries for the lifetime of the process.
TritonJITFunction& kai_decode_pack() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_pack_lhs_qai8dxp_bf16_kernel");
  return function;
}

TritonJITFunction& kai_decode_matrix() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_w8_qai8dxp_decode_sdot_kernel");
  return function;
}

TritonJITFunction& kai_decode_stealing_matrix() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_w8_qai8dxp_decode_stealing_sdot_kernel");
  return function;
}

TritonJITFunction& kai_prefill_pack() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_pack_lhs_qai8dxp_bf16_mr4_kernel");
  return function;
}

TritonJITFunction& kai_prefill_matrix() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_w8_qai8dxp_prefill_i8mm_kernel");
  return function;
}

TritonJITFunction& kai_prefill_stealing_matrix() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_w8_qai8dxp_prefill_stealing_n_i8mm_kernel");
  return function;
}

TritonJITFunction& kai_prefill_short_tail_matrix() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_w8_qai8dxp_prefill_short_tail_kernel");
  return function;
}

TritonJITFunction& kai_prefill_m12_matrix() {
  static TritonJITFunction& function = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_w8_qai8dxp_prefill_m12_kernel");
  return function;
}

int32_t parse_steal_chunk(const char* variable, int32_t fallback) {
  const char* configured = std::getenv(variable);
  if (configured == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  const long parsed = std::strtol(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 &&
                  parsed <= 1024,
              variable, " must be in [1,1024]");
  return static_cast<int32_t>(parsed);
}

int32_t kai_decode_steal_chunk(int64_t work) {
  if (work < 64 * 1024 * 1024) {
    return parse_steal_chunk("FLAGGEMS_W8_BODY_STEAL_CHUNK", 8);
  }
  return parse_steal_chunk("FLAGGEMS_W8_STEAL_CHUNK", 64);
}

bool use_kai_stealing_prefill() {
  const char* configured = std::getenv("FLAGGEMS_W8_STEALING_PREFILL");
  if (configured == nullptr) {
    return false;
  }
  return std::strcmp(configured, "0") != 0 &&
      std::strcmp(configured, "false") != 0 &&
      std::strcmp(configured, "off") != 0;
}

int32_t kai_prefill_steal_chunk() {
  const char* configured = std::getenv("FLAGGEMS_W8_PREFILL_STEAL_CHUNK");
  if (configured == nullptr) {
    return 2;
  }
  char* end = nullptr;
  const long parsed = std::strtol(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 &&
                  parsed <= 32,
              "FLAGGEMS_W8_PREFILL_STEAL_CHUNK must be in [1,32]");
  return static_cast<int32_t>(parsed);
}

class ScopedW8PrefillThreads {
 public:
  ScopedW8PrefillThreads() {
    const char* configured = std::getenv("FLAGGEMS_W8_PREFILL_THREADS");
    if (configured == nullptr) {
      return;
    }
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 &&
                    parsed <= 256,
                "FLAGGEMS_W8_PREFILL_THREADS must be in [1,256]");
    previous_ = omp_get_max_threads();
    omp_set_num_threads(static_cast<int>(parsed));
    active_ = true;
  }

  ~ScopedW8PrefillThreads() {
    if (active_) {
      omp_set_num_threads(previous_);
    }
  }

  ScopedW8PrefillThreads(const ScopedW8PrefillThreads&) = delete;
  ScopedW8PrefillThreads& operator=(const ScopedW8PrefillThreads&) = delete;

 private:
  int previous_ = 0;
  bool active_ = false;
};

at::Tensor run_decode(const at::Tensor& input,
                      const at::Tensor& decode_rhs,
                      const at::Tensor& scale,
                      int64_t n,
                      int64_t k) {
  at::Tensor x_q = at::empty({k}, input.options().dtype(at::kChar));
  at::Tensor x_scale = at::empty({1}, input.options().dtype(at::kFloat));
  at::Tensor output = at::empty(output_shape(input, n), input.options());
  TritonJITFunction& quantize = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_quantize_bf16_w8_vllm_trunc_kernel");
  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_w8_decode_sdot_kernel");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  constexpr int32_t block_k = 16;
  const int32_t block_n = n32 >= 32768 ? 32 : 64;
  constexpr int32_t unroll = 2;
  quantize(nullptr,
           1,
           1,
           1,
           1,
           1,
           input,
           x_q,
           x_scale,
           k32,
           block_k);
  matrix(nullptr,
         static_cast<unsigned int>(n32 / block_n),
         1,
         1,
         1,
         1,
         x_q,
         x_scale,
         decode_rhs,
         scale,
         output,
         k32,
         n32,
         block_n,
         unroll,
         false);
  return output;
}

at::Tensor run_prefill(const at::Tensor& input,
                       const at::Tensor& prefill_rhs,
                       int64_t n,
                       int64_t k,
                       int64_t m) {
  int32_t m_kernel;
  if (m <= 4) {
    m_kernel = 4;
  } else if (m <= 8) {
    m_kernel = 8;
  } else if (m <= 12) {
    m_kernel = 12;
  } else {
    m_kernel = checked_i32(((m + 15) / 16) * 16, "W8 padded M");
  }
  const int64_t panel_stride = 4 * k + 16;
  at::Tensor lhs = at::empty(
      {(static_cast<int64_t>(m_kernel) / 4) * panel_stride},
      input.options().dtype(at::kChar));
  at::Tensor output = at::empty({m_kernel, n}, input.options());
  at::Tensor input_2d = input.view({m, k});
  TritonJITFunction& pack = TritonJITFunction::get_instance(
      W8_KERNEL_SOURCE, "_pack_lhs_w8_i8mm_kai_vllm_trunc_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const bool full_rows = m32 == m_kernel;
  pack(nullptr,
       static_cast<unsigned int>(m_kernel),
       1,
       1,
       1,
       1,
       input_2d,
       lhs,
       m32,
       k32,
       k32,
       full_rows);

  if (m_kernel <= 8) {
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        W8_KERNEL_SOURCE, "_w8_prefill_i8mm_kai_short_tail_kernel");
    matrix(nullptr,
           1,
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs,
           prefill_rhs,
           output,
           n32,
           k32,
           m_kernel);
  } else if (m_kernel == 12) {
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        W8_KERNEL_SOURCE, "_w8_prefill_i8mm_kai_m12_kernel");
    matrix(nullptr,
           1,
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs,
           prefill_rhs,
           output,
           n32,
           k32);
  } else {
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        W8_KERNEL_SOURCE, "_w8_prefill_i8mm_kai_kernel");
    matrix(nullptr,
           static_cast<unsigned int>(m_kernel / 16),
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs,
           prefill_rhs,
           output,
           n32,
           k32,
           16);
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor w8_linear_cpu(const at::Tensor& input,
                         const at::Tensor& decode_rhs,
                         const at::Tensor& prefill_rhs,
                         const at::Tensor& scale,
                         int64_t n,
                         int64_t k) {
  validate(input, decode_rhs, prefill_rhs, scale, n, k);
  const int64_t m = input.numel() / k;
  return m == 1 ? run_decode(input, decode_rhs, scale, n, k)
                : run_prefill(input, prefill_rhs, n, k, m);
}

at::Tensor run_kai_decode(const at::Tensor& input,
                          const at::Tensor& rhs,
                          int64_t n,
                          int64_t k) {
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t partitions = kai_decode_partitions(k, n);
  const int64_t work = k * n;
  const bool stealing = use_kai_stealing_decode() &&
      work >= kai_stealing_min_work() &&
      partitions == at::get_num_threads();
  const int64_t lhs_bytes = k + 8;
  const int64_t output_bytes = n * input.element_size();
  // Keep the returned tensor at storage offset zero and place the short-lived
  // activation pack and optional work counter on cache-line boundaries in the
  // same allocation. This avoids one allocator round-trip per stealing GEMV.
  const int64_t lhs_offset = 64 * ((output_bytes + 63) / 64);
  const int64_t counter_offset =
      64 * ((lhs_offset + lhs_bytes + 63) / 64);
  const int64_t storage_bytes =
      stealing ? counter_offset + 64 : lhs_offset + lhs_bytes;
  at::Tensor storage = at::empty(
      {storage_bytes}, input.options().dtype(at::kByte));
  at::Tensor output = storage.narrow(0, 0, output_bytes)
                          .view(at::kBFloat16)
                          .view(output_shape(input, n));
  at::Tensor lhs =
      storage.narrow(0, lhs_offset, lhs_bytes).view(at::kChar);
  TritonJITFunction& pack = kai_decode_pack();
  TritonJITFunction& matrix =
      stealing ? kai_decode_stealing_matrix() : kai_decode_matrix();
  pack(nullptr, 1, 1, 1, 1, 1, input, lhs, 1, k32, k32, 1);
  if (stealing) {
    at::Tensor counter = storage.narrow(0, counter_offset, 64).view(at::kInt);
    *counter.data_ptr<int32_t>() = 0;
    matrix(nullptr,
           static_cast<unsigned int>(partitions),
           1,
           1,
           1,
           1,
           lhs,
           rhs,
           output,
           counter,
           0,
           n32 / 4,
           k32,
           n32,
           2,
           kai_decode_steal_chunk(work));
  } else {
    matrix(nullptr,
           static_cast<unsigned int>(partitions),
           1,
           1,
           1,
           1,
           lhs,
           rhs,
           output,
           0,
           n32 / 4,
           k32,
           n32,
           2);
  }
  return output;
}

int32_t tail_block(int32_t rows) {
  TORCH_CHECK(rows > 0 && rows <= 16,
              "exact-KAI W8 tail must contain 1..16 rows");
  return std::min<int32_t>(16, 4 * ((rows + 3) / 4));
}

at::Tensor run_kai_prefill(const at::Tensor& input,
                           const at::Tensor& rhs,
                           int64_t n,
                           int64_t k,
                           int64_t m) {
  ScopedW8PrefillThreads threads;
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t padded_m = 4 * ((m32 + 3) / 4);
  const int64_t lhs_panel_stride = 4 * k + 32;
  const int64_t lhs_bytes =
      (static_cast<int64_t>(padded_m) / 4) * lhs_panel_stride;
  at::Tensor lhs = at::empty({lhs_bytes}, input.options().dtype(at::kChar));
  at::Tensor output = at::empty({padded_m, n}, input.options());
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = kai_prefill_pack();
  pack(nullptr,
       static_cast<unsigned int>(padded_m / 4),
       1,
       1,
       1,
       1,
       input_2d,
       lhs,
       m32,
       k32,
       k32);

  const int32_t main_rows = (m32 / 16) * 16;
  if (main_rows > 0) {
    if (use_kai_stealing_prefill() && main_rows > 16) {
      TritonJITFunction& matrix = kai_prefill_stealing_matrix();
      const int32_t partitions = std::min<int32_t>(
          std::max(1, at::get_num_threads()), n32 / 4);
      at::Tensor counter = at::zeros({1}, input.options().dtype(at::kInt));
      matrix(nullptr,
             static_cast<unsigned int>(partitions),
             1,
             1,
             1,
             1,
             lhs,
             rhs,
             output,
             counter,
             n32,
             k32,
             main_rows / 16,
             kai_prefill_steal_chunk());
    } else {
      TritonJITFunction& matrix = kai_prefill_matrix();
      matrix(nullptr,
             static_cast<unsigned int>(main_rows / 16),
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs,
             rhs,
             output,
             n32,
             k32,
             16);
    }
  }

  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t block_m = tail_block(remaining);
    const int64_t lhs_offset =
        (static_cast<int64_t>(main_rows) / 4) * lhs_panel_stride;
    at::Tensor lhs_tail = lhs.narrow(0, lhs_offset, lhs_bytes - lhs_offset);
    at::Tensor output_tail = output.narrow(0, main_rows, padded_m - main_rows);
    if (block_m <= 8) {
      TritonJITFunction& matrix = kai_prefill_short_tail_matrix();
      matrix(nullptr,
             1,
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs_tail,
             rhs,
             output_tail,
             n32,
             k32,
             block_m);
    } else if (block_m == 12) {
      TritonJITFunction& matrix = kai_prefill_m12_matrix();
      matrix(nullptr,
             1,
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs_tail,
             rhs,
             output_tail,
             n32,
             k32);
    } else {
      TritonJITFunction& matrix = kai_prefill_matrix();
      matrix(nullptr,
             1,
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs_tail,
             rhs,
             output_tail,
             n32,
             k32,
             16);
    }
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor w8_linear_kai_cpu(const at::Tensor& input,
                             const at::Tensor& rhs,
                             int64_t n,
                             int64_t k) {
  validate_kai(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  return m == 1 ? run_kai_decode(input, rhs, n, k)
                : run_kai_prefill(input, rhs, n, k, m);
}

at::Tensor w8_linear_meta(const at::Tensor& input,
                          const at::Tensor&,
                          const at::Tensor&,
                          const at::Tensor&,
                          int64_t n,
                          int64_t) {
  std::vector<c10::SymInt> shape = input.sym_sizes().vec();
  TORCH_CHECK(!shape.empty(), "W8 input must have at least one dimension");
  shape.back() = c10::SymInt(n);
  return input.new_empty_symint(shape, input.options().dtype(at::kBFloat16));
}

at::Tensor w8_linear_kai_meta(const at::Tensor& input,
                              const at::Tensor&,
                              int64_t n,
                              int64_t) {
  std::vector<c10::SymInt> shape = input.sym_sizes().vec();
  TORCH_CHECK(!shape.empty(), "W8 input must have at least one dimension");
  shape.back() = c10::SymInt(n);
  return input.new_empty_symint(shape, input.options().dtype(at::kBFloat16));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(triton_jit_cpu, library) {
  library.def(
      "w8_linear(Tensor input, Tensor decode_rhs, Tensor prefill_rhs, "
      "Tensor scale, int n, int k) -> Tensor");
  library.def(
      "w8_linear_kai(Tensor input, Tensor rhs, int n, int k) -> Tensor");
}

TORCH_LIBRARY_IMPL(triton_jit_cpu, CPU, library) {
  library.impl("w8_linear", w8_linear_cpu);
  library.impl("w8_linear_kai", w8_linear_kai_cpu);
}

TORCH_LIBRARY_IMPL(triton_jit_cpu, Meta, library) {
  library.impl("w8_linear", w8_linear_meta);
  library.impl("w8_linear_kai", w8_linear_kai_meta);
}
