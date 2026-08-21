#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <omp.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "triton_jit/triton_jit_function.h"
#include "kernel_sources.h"

namespace {

using triton_jit::TritonJITFunction;

constexpr int64_t kBlockLength = 32;
constexpr int64_t kG128BlockLength = 128;

struct LaunchProfileRecord {
  uint64_t calls = 0;
  uint64_t elapsed_ns = 0;
};

std::mutex launch_profile_mutex;
std::map<std::string, LaunchProfileRecord> launch_profile_records;
std::atomic<bool> launch_profile_active {false};
thread_local std::chrono::steady_clock::time_point launch_profile_started;
thread_local std::string launch_profile_key;

void add_launch_profile_record(const std::string& key, uint64_t elapsed_ns) {
  std::lock_guard<std::mutex> lock(launch_profile_mutex);
  LaunchProfileRecord& record = launch_profile_records[key];
  ++record.calls;
  record.elapsed_ns += elapsed_ns;
}

class ScopedOpProfile {
 public:
  void start(std::string key) {
    key_ = std::move(key);
    started_ = std::chrono::steady_clock::now();
    active_ = true;
  }

  ~ScopedOpProfile() {
    if (!active_) {
      return;
    }
    const auto elapsed = std::chrono::steady_clock::now() - started_;
    add_launch_profile_record(
        key_, static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(elapsed).count()));
  }

 private:
  bool active_ = false;
  std::string key_;
  std::chrono::steady_clock::time_point started_;
};

void launch_profile_start() {
  triton_jit::clear_launch_hooks();
  {
    std::lock_guard<std::mutex> lock(launch_profile_mutex);
    launch_profile_records.clear();
  }
  launch_profile_active.store(true, std::memory_order_release);
  triton_jit::set_launch_enter_hook(
      [](const triton_jit::LaunchMetadata& metadata) {
        launch_profile_key = metadata.kernel_name + "@" +
            std::to_string(metadata.grid_x) + "x" +
            std::to_string(metadata.grid_y) + "x" +
            std::to_string(metadata.grid_z);
        launch_profile_started = std::chrono::steady_clock::now();
      });
  triton_jit::set_launch_exit_hook(
      [](const triton_jit::LaunchMetadata&) {
        const auto elapsed = std::chrono::steady_clock::now() -
            launch_profile_started;
        const auto elapsed_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(elapsed).count();
        add_launch_profile_record(
            launch_profile_key, static_cast<uint64_t>(elapsed_ns));
      });
}

std::string launch_profile_stop() {
  triton_jit::clear_launch_hooks();
  launch_profile_active.store(false, std::memory_order_release);
  std::lock_guard<std::mutex> lock(launch_profile_mutex);
  std::ostringstream output;
  output << "{\"kernels\":[";
  bool first = true;
  for (const auto& [key, record] : launch_profile_records) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << "{\"key\":\"" << key << "\",\"calls\":"
           << record.calls << ",\"elapsed_ns\":" << record.elapsed_ns
           << '}';
  }
  output << "]}";
  return output.str();
}

std::vector<int64_t> output_shape(const at::Tensor& input, int64_t n) {
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  TORCH_CHECK(!shape.empty(), "Q4 input must have at least one dimension");
  shape.back() = n;
  return shape;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0;
       --index) {
    strides[index] = strides[index + 1] * shape[index + 1];
  }
  return strides;
}

int32_t checked_i32(int64_t value, const char* name) {
  TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() &&
                  value <= std::numeric_limits<int32_t>::max(),
              name, " does not fit the Triton CPU i32 ABI");
  return static_cast<int32_t>(value);
}

int32_t decode_partitions(int64_t k, int64_t n) {
  if (k * n < 2 * 1024 * 1024) {
    return 1;
  }
  const int32_t threads = std::max(1, at::get_num_threads());
  if (const char* configured = std::getenv("FLAGGEMS_Q4_DECODE_PARTITIONS")) {
    if (std::string(configured) == "auto") {
      // Continue into the measured Apple shape selector below.
    } else {
      char* end = nullptr;
      const long parsed = std::strtol(configured, &end, 10);
      TORCH_CHECK(
          end != configured && *end == '\0' && parsed > 0,
          "FLAGGEMS_Q4_DECODE_PARTITIONS must be 'auto' or a positive "
          "integer");
      return std::min<int32_t>(
          std::min<int32_t>(threads, checked_i32(parsed, "decode partitions")),
          checked_i32(n / 64, "N/64"));
    }
  }
  int32_t default_partitions = threads;
#if defined(__APPLE__)
  // Interleaved Qwen3.6-27B M5 measurements with the compact production
  // layout show that only the medium-width projections benefit from fewer
  // workers: the attention QKVZ shape (N/K ~= 2.8) favors 10 and the GDN
  // input shape (N/K ~= 3.2) favors 12.  Gate/up, down, and output projections
  // use all available workers.  Ratios keep the selector useful for the same
  // architecture while the final min with `threads` is safe on smaller Macs.
  if (n >= 4 * k) {
    // Keep all workers for the very wide joined gate/up projection.
  } else if (n >= 3 * k) {
    default_partitions = std::min<int32_t>(default_partitions, 12);
  } else if (n >= 2 * k) {
    default_partitions = std::min<int32_t>(default_partitions, 10);
  }
#endif
  return std::min<int32_t>(default_partitions,
                           checked_i32(n / 64, "N/64"));
}

int32_t decode_unroll(int64_t k, int64_t n) {
  const char* configured = std::getenv("FLAGGEMS_Q4_DECODE_UNROLL");
  if (configured == nullptr) {
    // The wide joined gate/up projection has enough work per partition to
    // amortize a two-group unroll.  Narrow projections and the K6144 down
    // projection are neutral or regress because the larger body competes
    // with activation-pack state and cache bandwidth.
    return n >= 4 * k ? 2 : 1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' &&
                  (parsed == 1 || parsed == 2 || parsed == 4),
              "FLAGGEMS_Q4_DECODE_UNROLL must be 1, 2, or 4");
  return static_cast<int32_t>(parsed);
}

int32_t g128_decode_unroll(int64_t k, int64_t n) {
  if (const char* configured =
          std::getenv("FLAGGEMS_ARM_Q4_G128_DECODE_UNROLL")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' &&
                    (parsed == 1 || parsed == 2 || parsed == 4),
                "FLAGGEMS_ARM_Q4_G128_DECODE_UNROLL must be 1, 2, or 4");
    return static_cast<int32_t>(parsed);
  }
  // Keep the existing global control useful for same-engine A/B tests.
  if (std::getenv("FLAGGEMS_Q4_DECODE_UNROLL") != nullptr) {
    return decode_unroll(k, n);
  }
  // End-to-end Qwen3.6 measurements favor the smaller body once all 304
  // projections and their heterogeneous-core tail latency are included.
  return 1;
}

bool use_shared_decode_pack() {
  const char* configured = std::getenv("FLAGGEMS_Q4_DECODE_SHARED_PACK");
  if (configured == nullptr) {
    return false;
  }
  const std::string value(configured);
  TORCH_CHECK(value == "0" || value == "1" || value == "false" ||
                  value == "true" || value == "off" || value == "on",
              "FLAGGEMS_Q4_DECODE_SHARED_PACK must be a boolean");
  return value == "1" || value == "true" || value == "on";
}

bool use_g128_stealing_decode() {
  const char* configured =
      std::getenv("FLAGGEMS_ARM_Q4_G128_STEALING_DECODE");
  if (configured == nullptr) {
    return false;
  }
  const std::string value(configured);
  TORCH_CHECK(value == "0" || value == "1" || value == "false" ||
                  value == "true" || value == "off" || value == "on",
              "FLAGGEMS_ARM_Q4_G128_STEALING_DECODE must be a boolean");
  return value == "1" || value == "true" || value == "on";
}

bool use_g128_swiglu_stealing_decode() {
  const char* configured =
      std::getenv("FLAGGEMS_ARM_Q4_G128_SWIGLU_STEALING");
  if (configured == nullptr) {
    return false;
  }
  const std::string value(configured);
  TORCH_CHECK(value == "0" || value == "1" || value == "false" ||
                  value == "true" || value == "off" || value == "on",
              "FLAGGEMS_ARM_Q4_G128_SWIGLU_STEALING must be a boolean");
  return value == "1" || value == "true" || value == "on";
}

int64_t g128_stealing_min_work() {
  const char* configured =
      std::getenv("FLAGGEMS_ARM_Q4_G128_STEALING_MIN_WORK");
  if (configured == nullptr) {
    return 64 * 1024 * 1024;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' && parsed >= 0,
              "FLAGGEMS_ARM_Q4_G128_STEALING_MIN_WORK must be non-negative");
  return static_cast<int64_t>(parsed);
}

bool use_weighted_decode() {
  const char* configured = std::getenv("FLAGGEMS_Q4_WEIGHTED_DECODE");
  if (configured == nullptr) {
    return false;
  }
  const std::string value(configured);
  TORCH_CHECK(value == "0" || value == "1" || value == "false" ||
                  value == "true" || value == "off" || value == "on",
              "FLAGGEMS_Q4_WEIGHTED_DECODE must be a boolean");
  return value == "1" || value == "true" || value == "on";
}

bool use_stealing_decode() {
  const char* configured = std::getenv("FLAGGEMS_Q4_STEALING_DECODE");
  if (configured == nullptr) {
    return false;
  }
  const std::string value(configured);
  TORCH_CHECK(value == "0" || value == "1" || value == "false" ||
                  value == "true" || value == "off" || value == "on",
              "FLAGGEMS_Q4_STEALING_DECODE must be a boolean");
  return value == "1" || value == "true" || value == "on";
}

int32_t decode_steal_chunk() {
  const char* configured = std::getenv("FLAGGEMS_Q4_STEAL_CHUNK");
  if (configured == nullptr) {
    return 32;
  }
  char* end = nullptr;
  const long parsed = std::strtol(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 &&
                  parsed <= 1024,
              "FLAGGEMS_Q4_STEAL_CHUNK must be an integer in [1, 1024]");
  return static_cast<int32_t>(parsed);
}

bool use_g32_prefill_n8() {
  const char* configured = std::getenv("FLAGGEMS_Q4_G32_PREFILL_N8");
  if (configured == nullptr) {
    return false;
  }
  const std::string value(configured);
  TORCH_CHECK(value == "0" || value == "1" || value == "false" ||
                  value == "true" || value == "off" || value == "on",
              "FLAGGEMS_Q4_G32_PREFILL_N8 must be a boolean");
  return value == "1" || value == "true" || value == "on";
}

int32_t tail_block(int64_t rows) {
  TORCH_CHECK(rows > 0 && rows <= 16, "invalid Q4 prefill tail: ", rows);
  return static_cast<int32_t>(std::min<int64_t>(16, 4 * ((rows + 3) / 4)));
}

int32_t g32_prefill_block(int64_t rows) {
  if (const char* configured =
          std::getenv("FLAGGEMS_Q4_G32_PREFILL_BLOCK_M")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' &&
                    (parsed == 4 || parsed == 8 || parsed == 12 ||
                     parsed == 16),
                "FLAGGEMS_Q4_G32_PREFILL_BLOCK_M must be 4, 8, 12, or 16");
    return static_cast<int32_t>(parsed);
  }
  // Qwen3.6's five production G32 shapes all favor M16 for a 512-token
  // prefill on M5 Pro (3-10% per projection, bit-exact versus M8).  Keep the
  // choice overrideable because the register-pressure tradeoff must be
  // independently checked on the M4 Pro target.
  return 16;
}

int32_t g128_prefill_block(int64_t rows) {
  if (const char* configured =
          std::getenv("FLAGGEMS_ARM_Q4_G128_PREFILL_BLOCK_M")) {
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' &&
                    (parsed == 4 || parsed == 8 || parsed == 12 ||
                     parsed == 16),
                "FLAGGEMS_ARM_Q4_G128_PREFILL_BLOCK_M must be 4, 8, 12, or 16");
    return static_cast<int32_t>(parsed);
  }
  if (rows <= 16) {
    return 16;
  }
  if (rows >= 96 || rows % 12 == 0) {
    return 12;
  }
  if (rows % 16 == 0) {
    return 16;
  }
  if (rows % 8 == 0) {
    return 8;
  }
  return 12;
}

int32_t g128_prefill_subgroup_unroll() {
  const char* configured =
      std::getenv("FLAGGEMS_ARM_Q4_G128_PREFILL_SUBGROUP_UNROLL");
  if (configured == nullptr) {
    return 1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' &&
                  (parsed == 1 || parsed == 2 || parsed == 4),
              "FLAGGEMS_ARM_Q4_G128_PREFILL_SUBGROUP_UNROLL must be 1, 2, or 4");
  return static_cast<int32_t>(parsed);
}

bool use_g128_stealing_prefill() {
  const char* configured =
      std::getenv("FLAGGEMS_ARM_Q4_G128_STEALING_PREFILL");
  if (configured == nullptr) {
    return false;
  }
  return std::strcmp(configured, "0") != 0 &&
      std::strcmp(configured, "false") != 0 &&
      std::strcmp(configured, "off") != 0;
}

int32_t g128_steal_chunk() {
  const char* configured =
      std::getenv("FLAGGEMS_ARM_Q4_G128_STEAL_CHUNK");
  if (configured == nullptr) {
    return 2;
  }
  char* end = nullptr;
  const long parsed = std::strtol(configured, &end, 10);
  TORCH_CHECK(end != configured && *end == '\0' &&
                  (parsed == 1 || parsed == 2 || parsed == 4 ||
                   parsed == 8 || parsed == 16 || parsed == 32),
              "FLAGGEMS_ARM_Q4_G128_STEAL_CHUNK must be "
              "1, 2, 4, 8, 16, or 32");
  return static_cast<int32_t>(parsed);
}

class ScopedPrefillThreads {
 public:
  ScopedPrefillThreads() {
    const char* configured = std::getenv("FLAGGEMS_Q4_PREFILL_THREADS");
    if (configured == nullptr) {
      return;
    }
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 &&
                    parsed <= 256,
                "FLAGGEMS_Q4_PREFILL_THREADS must be in [1,256]");
    previous_ = omp_get_max_threads();
    omp_set_num_threads(static_cast<int>(parsed));
    active_ = true;
  }

  ~ScopedPrefillThreads() {
    if (active_) {
      omp_set_num_threads(previous_);
    }
  }

  ScopedPrefillThreads(const ScopedPrefillThreads&) = delete;
  ScopedPrefillThreads& operator=(const ScopedPrefillThreads&) = delete;

 private:
  int previous_ = 0;
  bool active_ = false;
};

void validate(const at::Tensor& input,
              const at::Tensor& rhs,
              int64_t n,
              int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "libtriton_jit Q4 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "libtriton_jit Q4 requires BF16 activation input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "libtriton_jit Q4 requires a contiguous UINT8 RHS blob");
  TORCH_CHECK(input.is_contiguous(),
              "libtriton_jit Q4 requires contiguous activation input");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid Q4 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kBlockLength == 0,
              "Q4 dimensions require N%4=0 and K%32=0");
  const int64_t expected = (n / 4) * (k / kBlockLength) * 72;
  TORCH_CHECK(rhs.numel() == expected, "invalid packed Q4 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

void validate_g128(const at::Tensor& input,
                   const at::Tensor& rhs,
                   int64_t n,
                   int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "libtriton_jit G128 Q4 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "libtriton_jit G128 Q4 requires BF16 activation input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "libtriton_jit G128 Q4 requires a contiguous UINT8 RHS blob");
  TORCH_CHECK(input.is_contiguous(),
              "libtriton_jit G128 Q4 requires contiguous activation input");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid G128 Q4 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kG128BlockLength == 0,
              "G128 Q4 dimensions require N%4=0 and K%128=0");
  const int64_t expected =
      (n / 4) * ((k / kG128BlockLength) * 264 + 16);
  TORCH_CHECK(rhs.numel() == expected,
              "invalid packed G128 Q4 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

void validate_g32_asym(const at::Tensor& input,
                       const at::Tensor& rhs,
                       int64_t n,
                       int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "libtriton_jit asymmetric G32 Q4 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
              "asymmetric G32 Q4 requires contiguous BF16 input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "asymmetric G32 Q4 requires a contiguous UINT8 RHS blob");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid asymmetric G32 Q4 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kBlockLength == 0,
              "asymmetric G32 Q4 dimensions require N%4=0 and K%32=0");
  const int64_t expected = (n / 4) * (k / kBlockLength) * 80;
  TORCH_CHECK(rhs.numel() == expected,
              "invalid packed asymmetric G32 Q4 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

void validate_g32_asym_compact(const at::Tensor& input,
                               const at::Tensor& rhs,
                               int64_t n,
                               int64_t k) {
  TORCH_CHECK(input.device().is_cpu() && rhs.device().is_cpu(),
              "compact asymmetric G32 Q4 supports CPU tensors only");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
              "compact asymmetric G32 Q4 requires contiguous BF16 input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "compact asymmetric G32 Q4 requires a contiguous UINT8 RHS");
  TORCH_CHECK(input.dim() > 0 && input.numel() > 0 && input.size(-1) == k,
              "invalid compact asymmetric G32 Q4 input shape");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kBlockLength == 0,
              "compact asymmetric G32 Q4 requires N%4=0 and K%32=0");
  const int64_t expected =
      (n / 4) * ((k / kBlockLength) * 72 + 16);
  TORCH_CHECK(rhs.numel() == expected,
              "invalid compact asymmetric G32 Q4 RHS byte count");
  checked_i32(n, "N");
  checked_i32(k, "K");
}

at::Tensor run_decode(const at::Tensor& input,
                      const at::Tensor& rhs,
                      int64_t n,
                      int64_t k,
                      int64_t m) {
  const int32_t partitions = decode_partitions(k, n);
  const int64_t scratch_bytes =
      m * partitions * (k / kBlockLength) * 34;
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned Q4 workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  const std::vector<int64_t> shape = output_shape(input, n);
  at::Tensor output = storage.as_strided(
      shape, contiguous_strides(shape), scratch_bytes / 2);

  TritonJITFunction& kernel = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_fused_decode_sdot_kai_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t output_offset = checked_i32(scratch_bytes, "workspace");
  const int32_t range_begin = 0;
  const int32_t range_end = n32 / 4;
  const int32_t unroll = k >= 4096 ? 4 : 1;
  kernel(nullptr,
         static_cast<unsigned int>(m32),
         static_cast<unsigned int>(partitions),
         1,
         1,
         1,
         input,
         storage,
         rhs,
         output_offset,
         k32,
         range_begin,
         range_end,
         k32,
         n32,
         unroll);
  return output;
}

at::Tensor run_prefill(const at::Tensor& input,
                       const at::Tensor& rhs,
                       int64_t n,
                       int64_t k,
                       int64_t m) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t groups = k / kBlockLength;
  const int64_t lhs_bytes = (padded_m / 4) * groups * 136;
  at::Tensor lhs_blob = at::empty({lhs_bytes}, input.options().dtype(at::kByte));
  at::Tensor lhs_scale = lhs_blob.view(at::kHalf);
  at::Tensor lhs_data = lhs_blob.view(at::kChar);
  at::Tensor rhs_scale = rhs.view(at::kHalf);
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_pack_lhs_qsi8d32p_panel4_scalar_kernel");
  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_i8mm_kai_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t full_panels = m32 / 4;
  if (full_panels > 0) {
    pack(nullptr,
         static_cast<unsigned int>(full_panels),
         1,
         1,
         1,
         1,
         input_2d,
         lhs_scale,
         lhs_data,
         m32,
         k32,
         k32,
         true);
  }
  const int32_t tail_rows = m32 - full_panels * 4;
  if (tail_rows > 0) {
    const int64_t lhs_offset = static_cast<int64_t>(full_panels) * groups * 136;
    at::Tensor input_tail = input_2d.narrow(0, full_panels * 4, tail_rows);
    at::Tensor lhs_tail = lhs_blob.narrow(0, lhs_offset, lhs_bytes - lhs_offset);
    pack(nullptr,
         1,
         1,
         1,
         1,
         1,
         input_tail,
         lhs_tail.view(at::kHalf),
         lhs_tail.view(at::kChar),
         tail_rows,
         k32,
         k32,
         false);
  }

  at::Tensor output = at::empty({padded_m, n}, input.options());
  const int32_t main_rows = (m32 / 16) * 16;
  const int32_t main_tiles = main_rows / 16;
  if (main_tiles > 0) {
    matrix(nullptr,
           static_cast<unsigned int>(main_tiles),
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs_data,
           lhs_scale,
           rhs,
           rhs_scale,
           output,
           n32,
           k32,
           16);
  }
  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t block_m = tail_block(remaining);
    const int64_t lhs_offset = (static_cast<int64_t>(main_rows) / 4) * groups * 136;
    at::Tensor lhs_tail = lhs_blob.narrow(0, lhs_offset, lhs_bytes - lhs_offset);
    at::Tensor output_tail = output.narrow(0, main_rows, padded_m - main_rows);
    matrix(nullptr,
           1,
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs_tail.view(at::kChar),
           lhs_tail.view(at::kHalf),
           rhs,
           rhs_scale,
           output_tail,
           n32,
           k32,
           block_m);
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor q4_linear_cpu(const at::Tensor& input,
                         const at::Tensor& rhs,
                         int64_t n,
                         int64_t k) {
  validate(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  return m < 4 ? run_decode(input, rhs, n, k, m)
               : run_prefill(input, rhs, n, k, m);
}

at::Tensor run_g32_asym_decode(const at::Tensor& input,
                               const at::Tensor& rhs,
                               int64_t n,
                               int64_t k,
                               int64_t m) {
  const int32_t partitions = decode_partitions(k, n);
  const int64_t scratch_bytes = m * partitions * (8 + k);
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned asymmetric G32 Q4 workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  at::Tensor output = storage.as_strided(
      output_shape(input, n), contiguous_strides(output_shape(input, n)),
      scratch_bytes / 2);

  TritonJITFunction& kernel = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_fused_decode_asym_g32_kai_sdot_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  kernel(nullptr,
         static_cast<unsigned int>(m32),
         static_cast<unsigned int>(partitions),
         1,
         1,
         1,
         input,
         storage,
         rhs,
         checked_i32(scratch_bytes, "workspace"),
         k32,
         0,
         n32 / 4,
         k32,
         n32,
         decode_unroll(k, n));
  return output;
}

at::Tensor run_g32_asym_compact_decode(const at::Tensor& input,
                                       const at::Tensor& rhs,
                                       int64_t n,
                                       int64_t k,
                                       int64_t m) {
  const int32_t partitions = decode_partitions(k, n);
  const bool shared_pack = use_shared_decode_pack();
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const bool stealing = !shared_pack && use_stealing_decode() &&
      k * n >= 64 * 1024 * 1024 && m32 == 1 &&
      partitions == at::get_num_threads();
  const int64_t activation_scratch_bytes =
      m * (shared_pack ? 1 : partitions) * (8 + k);
  constexpr int64_t kStealingCounterBytes = 64;
  const int64_t scratch_bytes = activation_scratch_bytes +
      (stealing ? kStealingCounterBytes : 0);
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned compact asymmetric G32 Q4 workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  at::Tensor output = storage.as_strided(
      output_shape(input, n), contiguous_strides(output_shape(input, n)),
      scratch_bytes / 2);

  if (shared_pack) {
    TritonJITFunction& pack = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE, "_pack_lhs_qai8dxp_asym_decode_kai_kernel");
    pack(nullptr,
         static_cast<unsigned int>(m32),
         1,
         1,
         1,
         1,
         input,
         storage,
         k32,
         k32);
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE,
        "_q4_decode_asym_g32_compact_sdot_flat_kernel");
    matrix(nullptr,
           static_cast<unsigned int>(m32 * partitions),
           1,
           1,
           1,
           1,
           storage,
           rhs,
           output,
           0,
           n32 / 4,
           k32,
           n32,
           partitions,
           decode_unroll(k, n));
  } else if (stealing) {
    auto* storage_bytes = reinterpret_cast<uint8_t*>(
        storage.data_ptr<at::BFloat16>());
    std::memset(storage_bytes + activation_scratch_bytes, 0,
                kStealingCounterBytes);
    TritonJITFunction& kernel = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE,
        "_q4_fused_decode_asym_g32_compact_stealing_kai_sdot_kernel");
    kernel(nullptr,
           static_cast<unsigned int>(m32),
           static_cast<unsigned int>(partitions),
           1,
           1,
           1,
           input,
           storage,
           rhs,
           checked_i32(scratch_bytes, "workspace"),
           k32,
           0,
           n32 / 4,
           k32,
           n32,
           decode_unroll(k, n),
           decode_steal_chunk());
  } else {
    // The occupancy probe and two extra barriers amortize on the three large
    // Qwen3.6 projections, but regress the 6144x5120 output projection.
    const bool weighted = use_weighted_decode() && k * n >= 64 * 1024 * 1024 &&
        m32 == 1 &&
        partitions == at::get_num_threads() && partitions <= 31;
    TritonJITFunction& kernel = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE,
        weighted
            ? "_q4_fused_decode_asym_g32_compact_weighted_kai_sdot_kernel"
            : "_q4_fused_decode_asym_g32_compact_kai_sdot_kernel");
    kernel(nullptr,
           static_cast<unsigned int>(m32),
           static_cast<unsigned int>(partitions),
           1,
           1,
           1,
           input,
           storage,
           rhs,
           checked_i32(scratch_bytes, "workspace"),
           k32,
           0,
           n32 / 4,
           k32,
           n32,
           decode_unroll(k, n));
  }
  return output;
}

at::Tensor q4_linear_g32_asym_compact_pair_cpu(
    const at::Tensor& input,
    const at::Tensor& rhs0,
    int64_t n0,
    const at::Tensor& rhs1,
    int64_t n1,
    int64_t k) {
  validate_g32_asym_compact(input, rhs0, n0, k);
  validate_g32_asym_compact(input, rhs1, n1, k);
  const int64_t m = input.numel() / k;
  TORCH_CHECK(m == 1,
              "compact G32 Q4 pair currently supports one decode row");

  const int32_t partitions = decode_partitions(k, n0 + n1);
  const bool stealing = use_stealing_decode() &&
      k * n0 >= 64 * 1024 * 1024 &&
      partitions == at::get_num_threads();
  const int64_t activation_scratch_bytes = m * partitions * (8 + k);
  constexpr int64_t kStealingCounterBytes = 64;
  const int64_t scratch_bytes = activation_scratch_bytes +
      (stealing ? kStealingCounterBytes : 0);
  const int64_t output_n = n0 + n1;
  const int64_t output_bytes = m * output_n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned compact asymmetric G32 Q4 pair workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  at::Tensor output = storage.as_strided(
      output_shape(input, output_n),
      contiguous_strides(output_shape(input, output_n)), scratch_bytes / 2);
  if (stealing) {
    auto* storage_bytes = reinterpret_cast<uint8_t*>(
        storage.data_ptr<at::BFloat16>());
    std::memset(storage_bytes + activation_scratch_bytes, 0,
                kStealingCounterBytes);
  }

  TritonJITFunction& kernel = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE,
      "_q4_fused_decode_asym_g32_compact_pair_kai_sdot_kernel");
  const int32_t n032 = checked_i32(n0, "N0");
  const int32_t n132 = checked_i32(n1, "N1");
  const int32_t k32 = checked_i32(k, "K");
  kernel(nullptr,
         1,
         static_cast<unsigned int>(partitions),
         1,
         1,
         1,
         input,
         storage,
         rhs0,
         rhs1,
         checked_i32(scratch_bytes, "workspace"),
         k32,
         k32,
         n032,
         n132,
         decode_unroll(k, n0),
         decode_unroll(k, n1),
         stealing,
         decode_steal_chunk());
  return output;
}

at::Tensor run_g32_asym_prefill(const at::Tensor& input,
                                const at::Tensor& rhs,
                                int64_t n,
                                int64_t k,
                                int64_t m) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t lhs_panel_stride = 32 + 4 * k;
  const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
  at::Tensor lhs = at::empty({lhs_bytes}, input.options().dtype(at::kByte));
  at::Tensor output = at::empty({padded_m, n}, input.options());
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_pack_lhs_qai8dxp_asym_panel4_kernel");
  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_asym_i8mm_kai_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
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

  // CIX A/B: one grid of paired M8 tiles beats the spilling M16 object,
  // whereas the joined M12 tail is faster than a separate M8 plus M4 launch.
  const int32_t main_block = g32_prefill_block(m32);
  const int32_t main_rows = (m32 / main_block) * main_block;
  if (main_rows > 0) {
    matrix(nullptr,
           static_cast<unsigned int>(main_rows / main_block),
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs,
           rhs,
           output,
           n32,
           k32,
           main_block,
           true);
  }
  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t block_m = tail_block(remaining);
    const int64_t lhs_offset =
        (static_cast<int64_t>(main_rows) / 4) * lhs_panel_stride;
    matrix(nullptr,
           1,
           static_cast<unsigned int>(n32 / 4),
           1,
           1,
           1,
           lhs.narrow(0, lhs_offset, lhs_bytes - lhs_offset),
           rhs,
           output.narrow(0, main_rows, padded_m - main_rows),
           n32,
           k32,
           block_m,
           true);
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

void run_g32_asym_compact_prefill_matrix(const at::Tensor& lhs,
                                         const at::Tensor& rhs,
                                         int64_t n,
                                         int64_t k,
                                         int64_t m,
                                         const at::Tensor& output,
                                         int64_t output_stride) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t lhs_panel_stride = 32 + 4 * k;
  const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
  TritonJITFunction& matrix_n4 = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_asym_g32_compact_i8mm_kai_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t output_stride32 = checked_i32(output_stride, "output stride");
  const bool n8 = use_g32_prefill_n8() && n32 % 8 == 0;
  const int32_t main_block = n8 ? 8 : g32_prefill_block(m32);
  const int32_t main_rows = (m32 / main_block) * main_block;
  if (main_rows > 0) {
    if (n8) {
      TritonJITFunction& matrix_n8 = TritonJITFunction::get_instance(
          Q4_KERNEL_SOURCE,
          "_q4_prefill_asym_g32_compact_i8mm_kai_m8n8_kernel");
      matrix_n8(nullptr,
                static_cast<unsigned int>(main_rows / 8),
                static_cast<unsigned int>(n32 / 8),
                1,
                1,
                1,
                lhs,
                rhs,
                output,
                output_stride32,
                k32);
    } else {
      matrix_n4(nullptr,
                static_cast<unsigned int>(main_rows / main_block),
                static_cast<unsigned int>(n32 / 4),
                1,
                1,
                1,
                lhs,
                rhs,
                output,
                output_stride32,
                k32,
                main_block);
    }
  }
  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t block_m = tail_block(remaining);
    const int64_t lhs_offset =
        (static_cast<int64_t>(main_rows) / 4) * lhs_panel_stride;
    matrix_n4(nullptr,
              1,
              static_cast<unsigned int>(n32 / 4),
              1,
              1,
              1,
              lhs.narrow(0, lhs_offset, lhs_bytes - lhs_offset),
              rhs,
              output.narrow(0, main_rows, padded_m - main_rows),
              output_stride32,
              k32,
              block_m);
  }
}

void run_g32_asym_compact_prefill_into(const at::Tensor& input,
                                       const at::Tensor& rhs,
                                       int64_t n,
                                       int64_t k,
                                       int64_t m,
                                       const at::Tensor& output,
                                       int64_t output_stride) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t lhs_panel_stride = 32 + 4 * k;
  const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
  at::Tensor lhs = at::empty({lhs_bytes}, input.options().dtype(at::kByte));
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_pack_lhs_qai8dxp_asym_panel4_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t k32 = checked_i32(k, "K");
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
  run_g32_asym_compact_prefill_matrix(
      lhs, rhs, n, k, m, output, output_stride);
}

at::Tensor run_g32_asym_compact_prefill(const at::Tensor& input,
                                        const at::Tensor& rhs,
                                        int64_t n,
                                        int64_t k,
                                        int64_t m) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  at::Tensor output = at::empty({padded_m, n}, input.options());
  int previous_threads = 0;
  bool restore_threads = false;
  if (n >= 1024) {
    if (const char* configured =
            std::getenv("FLAGGEMS_Q4_PREFILL_THREADS")) {
      char* end = nullptr;
      const long parsed = std::strtol(configured, &end, 10);
      TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 &&
                      parsed <= 256,
                  "FLAGGEMS_Q4_PREFILL_THREADS must be in [1,256]");
      previous_threads = omp_get_max_threads();
      omp_set_num_threads(static_cast<int>(parsed));
      restore_threads = true;
    }
  }
  try {
    run_g32_asym_compact_prefill_into(input, rhs, n, k, m, output, n);
  } catch (...) {
    if (restore_threads) {
      omp_set_num_threads(previous_threads);
    }
    throw;
  }
  if (restore_threads) {
    omp_set_num_threads(previous_threads);
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor q4_linear_g32_asym_cpu(const at::Tensor& input,
                                  const at::Tensor& rhs,
                                  int64_t n,
                                  int64_t k) {
  validate_g32_asym(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  return m < 4 ? run_g32_asym_decode(input, rhs, n, k, m)
               : run_g32_asym_prefill(input, rhs, n, k, m);
}

at::Tensor q4_linear_g32_asym_compact_cpu(const at::Tensor& input,
                                          const at::Tensor& rhs,
                                          int64_t n,
                                          int64_t k) {
  validate_g32_asym_compact(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  ScopedOpProfile profile;
  if (launch_profile_active.load(std::memory_order_acquire)) {
    profile.start("op:q4_compact@" + std::to_string(m) + "x" +
                  std::to_string(k) + "x" + std::to_string(n));
  }
  return m < 4 ? run_g32_asym_compact_decode(input, rhs, n, k, m)
               : run_g32_asym_compact_prefill(input, rhs, n, k, m);
}

at::Tensor q4_linear_g32_asym_compact_swiglu_cpu(
    const at::Tensor& joined,
    const at::Tensor& rhs,
    int64_t n,
    int64_t k) {
  TORCH_CHECK(joined.device().is_cpu() && rhs.device().is_cpu(),
              "compact G32 SwiGLU Q4 supports CPU tensors only");
  TORCH_CHECK(joined.scalar_type() == at::kBFloat16 && joined.is_contiguous(),
              "compact G32 SwiGLU Q4 requires contiguous BF16 input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "compact G32 SwiGLU Q4 requires contiguous UINT8 RHS");
  TORCH_CHECK(joined.dim() > 0 && joined.numel() > 0 &&
                  joined.size(-1) == 2 * k,
              "compact G32 SwiGLU Q4 requires a [...,2K] input");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % kBlockLength == 0,
              "compact G32 SwiGLU Q4 requires N%4=0 and K%32=0");
  const int64_t expected_rhs =
      (n / 4) * ((k / kBlockLength) * 72 + 16);
  TORCH_CHECK(rhs.numel() == expected_rhs,
              "invalid compact G32 SwiGLU Q4 RHS byte count");
  const int64_t m = joined.numel() / (2 * k);
  ScopedOpProfile profile;
  if (launch_profile_active.load(std::memory_order_acquire)) {
    profile.start("op:q4_compact_swiglu@" + std::to_string(m) + "x" +
                  std::to_string(k) + "x" + std::to_string(n));
  }
  if (m >= 4) {
    const int64_t padded_m = 4 * ((m + 3) / 4);
    const int64_t scratch_bytes = padded_m * k * 2;
    const int64_t lhs_panel_stride = 32 + 4 * k;
    const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
    at::Tensor storage = at::empty(
        {scratch_bytes + lhs_bytes}, joined.options().dtype(at::kByte));
    at::Tensor scratch =
        storage.narrow(0, 0, scratch_bytes).view(at::kBFloat16);
    at::Tensor lhs = storage.narrow(0, scratch_bytes, lhs_bytes);
    at::Tensor output = at::empty({padded_m, n}, joined.options());

    TritonJITFunction& pack = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE, "_q4_pack_swiglu_asym_panel4_kai_kernel");
    const int32_t m32 = checked_i32(m, "M");
    const int32_t k32 = checked_i32(k, "K");
    const int previous_threads = omp_get_max_threads();
    bool restore_threads = false;
    if (const char* configured =
            std::getenv("FLAGGEMS_Q4_PREFILL_THREADS")) {
      char* end = nullptr;
      const long parsed = std::strtol(configured, &end, 10);
      TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 &&
                      parsed <= 256,
                  "FLAGGEMS_Q4_PREFILL_THREADS must be in [1,256]");
      omp_set_num_threads(static_cast<int>(parsed));
      restore_threads = true;
    }
    try {
      pack(nullptr,
           static_cast<unsigned int>(padded_m / 4),
           1,
           1,
           1,
           1,
           joined.view({m, 2 * k}),
           scratch,
           lhs,
           m32,
           2 * k32,
           k32);
      run_g32_asym_compact_prefill_matrix(
          lhs, rhs, n, k, m, output, n);
    } catch (...) {
      if (restore_threads) {
        omp_set_num_threads(previous_threads);
      }
      throw;
    }
    if (restore_threads) {
      omp_set_num_threads(previous_threads);
    }
    return output.narrow(0, 0, m).view(output_shape(joined, n));
  }

  const int32_t partitions = decode_partitions(k, n);
  const int64_t scratch_bytes = m * k * 2;
  const int64_t lhs_bytes = m * (8 + k);
  const int64_t output_offset = scratch_bytes + lhs_bytes;
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK(output_offset % 2 == 0,
              "unaligned compact G32 SwiGLU workspace");
  at::Tensor storage = at::empty(
      {output_offset + output_bytes}, joined.options().dtype(at::kByte));
  at::Tensor scratch =
      storage.narrow(0, 0, scratch_bytes).view(at::kBFloat16);
  at::Tensor lhs = storage.narrow(0, scratch_bytes, lhs_bytes);
  at::Tensor output = storage.view(at::kBFloat16).as_strided(
      output_shape(joined, n), contiguous_strides(output_shape(joined, n)),
      output_offset / 2);

  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_pack_swiglu_asym_kai_kernel");
  pack(nullptr,
       static_cast<unsigned int>(m32),
       1,
       1,
       1,
       1,
       joined,
       scratch,
       lhs,
       2 * k32,
       k32,
       32);

  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_decode_asym_g32_compact_sdot_flat_kernel");
  matrix(nullptr,
         static_cast<unsigned int>(m32 * partitions),
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
         partitions,
         decode_unroll(k, n));
  return output;
}

at::Tensor q4_linear_g128_swiglu_cpu(const at::Tensor& joined,
                                      const at::Tensor& rhs,
                                      int64_t n,
                                      int64_t k) {
  TORCH_CHECK(joined.device().is_cpu() && rhs.device().is_cpu(),
              "G128 SwiGLU Q4 supports CPU tensors only");
  TORCH_CHECK(joined.scalar_type() == at::kBFloat16 && joined.is_contiguous(),
              "G128 SwiGLU Q4 requires contiguous BF16 input");
  TORCH_CHECK(rhs.scalar_type() == at::kByte && rhs.is_contiguous(),
              "G128 SwiGLU Q4 requires contiguous UINT8 RHS");
  TORCH_CHECK(joined.dim() > 0 && joined.numel() > 0 &&
                  joined.size(-1) == 2 * k,
              "G128 SwiGLU Q4 requires a [...,2K] input");
  TORCH_CHECK(n > 0 && k > 0 && n % 4 == 0 && k % 128 == 0,
              "G128 SwiGLU Q4 requires N%4=0 and K%128=0");
  const int64_t expected_rhs = (n / 4) * ((k / 128) * 264 + 16);
  TORCH_CHECK(rhs.numel() == expected_rhs,
              "invalid G128 SwiGLU Q4 RHS byte count");
  const int64_t m = joined.numel() / (2 * k);

  if (m >= 4) {
    const int64_t padded_m = 4 * ((m + 3) / 4);
    const int64_t scratch_bytes = padded_m * k * 2;
    const int64_t lhs_panel_stride = 32 + 4 * k;
    const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
    at::Tensor storage = at::empty(
        {scratch_bytes + lhs_bytes}, joined.options().dtype(at::kByte));
    at::Tensor scratch =
        storage.narrow(0, 0, scratch_bytes).view(at::kBFloat16);
    at::Tensor lhs = storage.narrow(0, scratch_bytes, lhs_bytes);
    at::Tensor output = at::empty({padded_m, n}, joined.options());
    const int32_t m32 = checked_i32(m, "M");
    const int32_t n32 = checked_i32(n, "N");
    const int32_t k32 = checked_i32(k, "K");
    TritonJITFunction& pack = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE, "_q4_pack_swiglu_asym_panel4_kai_kernel");
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE, "_q4_prefill_asym_g128_i8mm_kernel");
    pack(nullptr,
         static_cast<unsigned int>(padded_m / 4),
         1,
         1,
         1,
         1,
         joined.view({m, 2 * k}),
         scratch,
         lhs,
         m32,
         2 * k32,
         k32);
    const int32_t block_m = g128_prefill_block(m);
    const int32_t main_rows = (m32 / block_m) * block_m;
    if (main_rows > 0) {
      matrix(nullptr,
             static_cast<unsigned int>(main_rows / block_m),
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs,
             rhs,
             output,
             n32,
             k32,
             block_m,
             true,
             g128_prefill_subgroup_unroll());
    }
    const int32_t remaining = m32 - main_rows;
    if (remaining > 0) {
      const int32_t tail = tail_block(remaining);
      const int64_t lhs_offset =
          (static_cast<int64_t>(main_rows) / 4) * lhs_panel_stride;
      matrix(nullptr,
             1,
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs.narrow(0, lhs_offset, lhs_bytes - lhs_offset),
             rhs,
             output.narrow(0, main_rows, padded_m - main_rows),
             n32,
             k32,
             tail,
             true,
             g128_prefill_subgroup_unroll());
    }
    return output.narrow(0, 0, m).view(output_shape(joined, n));
  }

  const bool stealing = use_g128_swiglu_stealing_decode() &&
      k * n >= 64 * 1024 * 1024 && m == 1;
  const int32_t partitions = stealing
      ? checked_i32(at::get_num_threads(), "OpenMP thread count")
      : decode_partitions(k, n);
  const int64_t scratch_bytes = m * k * 2;
  const int64_t lhs_bytes = m * (8 + k);
  constexpr int64_t kStealingCounterBytes = 64;
  const int64_t output_offset = scratch_bytes + lhs_bytes +
      (stealing ? kStealingCounterBytes : 0);
  const int64_t output_bytes = m * n * 2;
  at::Tensor storage = at::empty(
      {output_offset + output_bytes}, joined.options().dtype(at::kByte));
  at::Tensor scratch =
      storage.narrow(0, 0, scratch_bytes).view(at::kBFloat16);
  at::Tensor lhs = storage.narrow(0, scratch_bytes, lhs_bytes);
  at::Tensor output = storage.view(at::kBFloat16).as_strided(
      output_shape(joined, n), contiguous_strides(output_shape(joined, n)),
      output_offset / 2);
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_pack_swiglu_asym_kai_kernel");
  pack(nullptr,
       static_cast<unsigned int>(m32),
       1,
       1,
       1,
       1,
       joined,
       scratch,
       lhs,
       2 * k32,
       k32,
       32);
  if (stealing) {
    auto* storage_bytes = storage.data_ptr<uint8_t>();
    std::memset(storage_bytes + output_offset - kStealingCounterBytes, 0,
                kStealingCounterBytes);
    at::Tensor counter = storage.narrow(
        0, output_offset - kStealingCounterBytes, kStealingCounterBytes);
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE,
        "_q4_decode_asym_g128_kai_shared_stealing_sdot_kernel");
    matrix(nullptr,
           static_cast<unsigned int>(m32),
           static_cast<unsigned int>(partitions),
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
           g128_decode_unroll(k, n),
           decode_steal_chunk());
  } else {
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE, "_q4_decode_asym_g128_kai_shared_sdot_kernel");
    matrix(nullptr,
           static_cast<unsigned int>(m32),
           static_cast<unsigned int>(partitions),
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
           g128_decode_unroll(k, n));
  }
  return output;
}

at::Tensor run_g128_decode(const at::Tensor& input,
                           const at::Tensor& rhs,
                           int64_t n,
                           int64_t k,
                           int64_t m) {
  const int32_t configured_partitions = decode_partitions(k, n);
  const bool shared_pack = use_shared_decode_pack();
  const bool stealing = !shared_pack && use_g128_stealing_decode() &&
      k * n >= g128_stealing_min_work() && m == 1;
  const int32_t partitions = stealing
      ? checked_i32(at::get_num_threads(), "OpenMP thread count")
      : configured_partitions;
  constexpr int64_t kStealingCounterBytes = 64;
  const int64_t activation_scratch_bytes =
      m * (shared_pack ? 1 : partitions) * (8 + k);
  const int64_t scratch_bytes =
      activation_scratch_bytes + (stealing ? kStealingCounterBytes : 0);
  const int64_t output_bytes = m * n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned G128 Q4 workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  const std::vector<int64_t> shape = output_shape(input, n);
  at::Tensor output = storage.as_strided(
      shape, contiguous_strides(shape), scratch_bytes / 2);

  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t output_offset = checked_i32(scratch_bytes, "workspace");
  const int32_t range_begin = 0;
  const int32_t range_end = n32 / 4;
  const int32_t unroll = g128_decode_unroll(k, n);
  if (shared_pack) {
    // Quantize the token once instead of repeating the same K-element scan in
    // every output partition.  Keep this behind the existing runtime switch:
    // the saved activation work must outweigh the extra launch at M < 4.
    TritonJITFunction& pack = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE, "_pack_lhs_qai8dxp_asym_decode_kai_kernel");
    pack(nullptr,
         static_cast<unsigned int>(m32),
         1,
         1,
         1,
         1,
         input,
         storage,
         k32,
         k32);
    TritonJITFunction& matrix = TritonJITFunction::get_instance(
        Q4_KERNEL_SOURCE, "_q4_decode_asym_g128_kai_shared_sdot_kernel");
    matrix(nullptr,
           static_cast<unsigned int>(m32),
           static_cast<unsigned int>(partitions),
           1,
           1,
           1,
           storage,
           rhs,
           output,
           range_begin,
           range_end,
           k32,
           n32,
           unroll);
  } else {
    if (stealing) {
      auto* storage_bytes = reinterpret_cast<uint8_t*>(
          storage.data_ptr<at::BFloat16>());
      std::memset(storage_bytes + activation_scratch_bytes, 0,
                  kStealingCounterBytes);
      TritonJITFunction& kernel = TritonJITFunction::get_instance(
          Q4_KERNEL_SOURCE,
          "_q4_fused_decode_asym_g128_stealing_kai_sdot_kernel");
      kernel(nullptr,
             static_cast<unsigned int>(m32),
             static_cast<unsigned int>(partitions),
             1,
             1,
             1,
             input,
             storage,
             rhs,
             output_offset,
             k32,
             range_begin,
             range_end,
             k32,
             n32,
             unroll,
             decode_steal_chunk());
    } else {
      TritonJITFunction& kernel = TritonJITFunction::get_instance(
          Q4_KERNEL_SOURCE, "_q4_fused_decode_asym_g128_kai_sdot_kernel");
      kernel(nullptr,
             static_cast<unsigned int>(m32),
             static_cast<unsigned int>(partitions),
             1,
             1,
             1,
             input,
             storage,
             rhs,
             output_offset,
             k32,
             range_begin,
             range_end,
             k32,
             n32,
             unroll);
    }
  }
  return output;
}

at::Tensor q4_linear_g128_pair_cpu(const at::Tensor& input,
                                    const at::Tensor& rhs0,
                                    int64_t n0,
                                    const at::Tensor& rhs1,
                                    int64_t n1,
                                    int64_t k) {
  validate_g128(input, rhs0, n0, k);
  validate_g128(input, rhs1, n1, k);
  const int64_t m = input.numel() / k;
  TORCH_CHECK(m == 1, "G128 Q4 pair currently supports one decode row");

  const int32_t partitions = decode_partitions(k, n0 + n1);
  const int64_t scratch_bytes = m * partitions * (8 + k);
  const int64_t output_n = n0 + n1;
  const int64_t output_bytes = m * output_n * 2;
  TORCH_CHECK((scratch_bytes + output_bytes) % 2 == 0,
              "unaligned G128 Q4 pair workspace");
  at::Tensor storage = at::empty(
      {(scratch_bytes + output_bytes) / 2}, input.options());
  at::Tensor output = storage.as_strided(
      output_shape(input, output_n),
      contiguous_strides(output_shape(input, output_n)), scratch_bytes / 2);

  TritonJITFunction& kernel = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE,
      "_q4_fused_decode_asym_g128_pair_kai_sdot_kernel");
  const int32_t n032 = checked_i32(n0, "N0");
  const int32_t n132 = checked_i32(n1, "N1");
  const int32_t k32 = checked_i32(k, "K");
  kernel(nullptr,
         1,
         static_cast<unsigned int>(partitions),
         1,
         1,
         1,
         input,
         storage,
         rhs0,
         rhs1,
         checked_i32(scratch_bytes, "workspace"),
         k32,
         k32,
         n032,
         n132,
         g128_decode_unroll(k, n0),
         g128_decode_unroll(k, n1));
  return output;
}

at::Tensor run_g128_prefill(const at::Tensor& input,
                            const at::Tensor& rhs,
                            int64_t n,
                            int64_t k,
                            int64_t m) {
  const int64_t padded_m = 4 * ((m + 3) / 4);
  const int64_t lhs_panel_stride = 32 + 4 * k;
  const int64_t lhs_bytes = (padded_m / 4) * lhs_panel_stride;
  at::Tensor lhs_blob = at::empty({lhs_bytes}, input.options().dtype(at::kByte));
  at::Tensor input_2d = input.view({m, k});

  TritonJITFunction& pack = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_pack_lhs_qai8dxp_asym_panel4_kernel");
  TritonJITFunction& matrix = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_asym_g128_i8mm_kernel");
  TritonJITFunction& matrix_stealing = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE,
      "_q4_prefill_asym_g128_stealing_n_i8mm_kernel");
  TritonJITFunction& matrix_m12_k32 = TritonJITFunction::get_instance(
      Q4_KERNEL_SOURCE, "_q4_prefill_asym_g128_i8mm_kai_m12_k32_kernel");
  const int32_t m32 = checked_i32(m, "M");
  const int32_t n32 = checked_i32(n, "N");
  const int32_t k32 = checked_i32(k, "K");
  const int32_t panels = checked_i32(padded_m / 4, "G128 panels");
  pack(nullptr,
       static_cast<unsigned int>(panels),
       1,
       1,
       1,
       1,
       input_2d,
       lhs_blob,
       m32,
       k32,
       k32);

  at::Tensor output = at::empty({padded_m, n}, input.options());
  const int32_t block_m = g128_prefill_block(m);
  const int32_t main_rows = (m32 / block_m) * block_m;
  const int32_t main_tiles = main_rows / block_m;
  if (main_tiles > 0) {
    if (use_g128_stealing_prefill() && main_tiles > 1) {
      const int32_t partitions = std::min<int32_t>(
          std::max(1, omp_get_max_threads()), n32 / 4);
      at::Tensor counter = at::zeros({1}, input.options().dtype(at::kInt));
      matrix_stealing(nullptr,
                      static_cast<unsigned int>(partitions),
                      1,
                      1,
                      1,
                      1,
                      lhs_blob,
                      rhs,
                      output,
                      counter,
                      n32,
                      k32,
                      block_m,
                      main_tiles,
                      true,
                      g128_prefill_subgroup_unroll(),
                      g128_steal_chunk());
    } else {
      matrix(nullptr,
             static_cast<unsigned int>(main_tiles),
             static_cast<unsigned int>(n32 / 4),
             1,
             1,
             1,
             lhs_blob,
             rhs,
             output,
             n32,
             k32,
             block_m,
             true,
             g128_prefill_subgroup_unroll());
    }
  }
  const int32_t remaining = m32 - main_rows;
  if (remaining > 0) {
    const int32_t tail = tail_block(remaining);
    const int64_t lhs_offset =
        (static_cast<int64_t>(main_rows) / 4) * lhs_panel_stride;
    at::Tensor lhs_tail = lhs_blob.narrow(0, lhs_offset, lhs_bytes - lhs_offset);
    at::Tensor output_tail = output.narrow(0, main_rows, padded_m - main_rows);
    if (tail == 12 && n >= 8192 && k <= 2048) {
      matrix_m12_k32(nullptr,
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
             tail,
             true,
             g128_prefill_subgroup_unroll());
    }
  }
  return output.narrow(0, 0, m).view(output_shape(input, n));
}

at::Tensor q4_linear_g128_cpu(const at::Tensor& input,
                              const at::Tensor& rhs,
                              int64_t n,
                              int64_t k) {
  validate_g128(input, rhs, n, k);
  const int64_t m = input.numel() / k;
  if (m < 4) {
    return run_g128_decode(input, rhs, n, k, m);
  }
  ScopedPrefillThreads threads;
  return run_g128_prefill(input, rhs, n, k, m);
}

at::Tensor q4_linear_meta(const at::Tensor& input,
                          const at::Tensor&,
                          int64_t n,
                          int64_t) {
  std::vector<c10::SymInt> shape = input.sym_sizes().vec();
  TORCH_CHECK(!shape.empty(), "Q4 input must have at least one dimension");
  shape.back() = c10::SymInt(n);
  return input.new_empty_symint(shape, input.options().dtype(at::kBFloat16));
}

at::Tensor q4_linear_pair_meta(const at::Tensor& input,
                               const at::Tensor& rhs0,
                               int64_t n0,
                               const at::Tensor&,
                               int64_t n1,
                               int64_t k) {
  return q4_linear_meta(input, rhs0, n0 + n1, k);
}

}  // namespace

TORCH_LIBRARY(triton_jit_cpu, library) {
  library.def("launch_profile_start() -> ()", &launch_profile_start);
  library.def("launch_profile_stop() -> str", &launch_profile_stop);
  library.def("q4_linear(Tensor input, Tensor rhs, int n, int k) -> Tensor");
  library.def(
      "q4_linear_g32_asym(Tensor input, Tensor rhs, int n, int k) -> Tensor");
  library.def(
      "q4_linear_g32_asym_compact(Tensor input, Tensor rhs, int n, int k) "
      "-> Tensor");
  library.def(
      "q4_linear_g32_asym_compact_pair(Tensor input, Tensor rhs0, int n0, "
      "Tensor rhs1, int n1, int k) -> Tensor");
  library.def(
      "q4_linear_g32_asym_compact_swiglu(Tensor joined, Tensor rhs, int n, "
      "int k) -> Tensor");
  library.def(
      "q4_linear_g128(Tensor input, Tensor rhs, int n, int k) -> Tensor");
  library.def(
      "q4_linear_g128_pair(Tensor input, Tensor rhs0, int n0, Tensor rhs1, "
      "int n1, int k) -> Tensor");
  library.def(
      "q4_linear_g128_swiglu(Tensor joined, Tensor rhs, int n, int k) "
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(triton_jit_cpu, CPU, library) {
  library.impl("q4_linear", q4_linear_cpu);
  library.impl("q4_linear_g32_asym", q4_linear_g32_asym_cpu);
  library.impl("q4_linear_g32_asym_compact",
               q4_linear_g32_asym_compact_cpu);
  library.impl("q4_linear_g32_asym_compact_pair",
               q4_linear_g32_asym_compact_pair_cpu);
  library.impl("q4_linear_g32_asym_compact_swiglu",
               q4_linear_g32_asym_compact_swiglu_cpu);
  library.impl("q4_linear_g128", q4_linear_g128_cpu);
  library.impl("q4_linear_g128_pair", q4_linear_g128_pair_cpu);
  library.impl("q4_linear_g128_swiglu", q4_linear_g128_swiglu_cpu);
}

TORCH_LIBRARY_IMPL(triton_jit_cpu, Meta, library) {
  library.impl("q4_linear", q4_linear_meta);
  library.impl("q4_linear_g32_asym", q4_linear_meta);
  library.impl("q4_linear_g32_asym_compact", q4_linear_meta);
  library.impl("q4_linear_g32_asym_compact_pair", q4_linear_pair_meta);
  library.impl("q4_linear_g32_asym_compact_swiglu", q4_linear_meta);
  library.impl("q4_linear_g128", q4_linear_meta);
  library.impl("q4_linear_g128_pair", q4_linear_pair_meta);
  library.impl("q4_linear_g128_swiglu", q4_linear_meta);
}
