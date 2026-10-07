// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>
#include "raw_dispatch.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "gdn_op.h"

namespace {

using BFloat16 = at::BFloat16;
using BFloatVec = at::vec::Vectorized<BFloat16>;
using FloatVec = at::vec::Vectorized<float>;

bool env_flag(const char* variable, bool default_value) {
  const char* configured = std::getenv(variable);
  if (configured == nullptr) {
    return default_value;
  }
  const std::string value(configured);
  return value != "0" && value != "false" && value != "off";
}

class ScopedGdnThreads {
 public:
  explicit ScopedGdnThreads(const char* variable) {
#if defined(_OPENMP)
    const char* configured = std::getenv(variable);
    if (configured == nullptr) {
      return;
    }
    char* end = nullptr;
    const long parsed = std::strtol(configured, &end, 10);
    TORCH_CHECK(end != configured && *end == '\0' && parsed > 0 && parsed <= 256,
                variable,
                " must be in [1,256]");
    previous_threads_ = omp_get_max_threads();
    omp_set_num_threads(static_cast<int>(parsed));
    restore_ = true;
#endif
  }

  ~ScopedGdnThreads() {
#if defined(_OPENMP)
    if (restore_) {
      omp_set_num_threads(previous_threads_);
    }
#endif
  }

  ScopedGdnThreads(const ScopedGdnThreads&) = delete;
  ScopedGdnThreads& operator=(const ScopedGdnThreads&) = delete;

 private:
  int previous_threads_ = 1;
  bool restore_ = false;
};

float softplus(float value, double threshold) {
  if (value > threshold) {
    return value;
  }
  if (value < -threshold) {
    return std::exp(value);
  }
  return std::log1p(std::exp(value));
}

float bf16_dot_scaled(
    const BFloat16* lhs, const BFloat16* rhs, int64_t size, float lhs_scale, float rhs_scale) {
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec lhs_scale_vec(lhs_scale);
  const FloatVec rhs_scale_vec(rhs_scale);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    const BFloatVec lhs_bf16 = BFloatVec::loadu(lhs + index);
    const BFloatVec rhs_bf16 = BFloatVec::loadu(rhs + index);
    FloatVec lhs0;
    FloatVec lhs1;
    FloatVec rhs0;
    FloatVec rhs1;
    std::tie(lhs0, lhs1) = at::vec::convert_to_float(lhs_bf16);
    std::tie(rhs0, rhs1) = at::vec::convert_to_float(rhs_bf16);
    sum0 = sum0 + lhs0 * lhs_scale_vec * rhs0 * rhs_scale_vec;
    sum1 = sum1 + lhs1 * lhs_scale_vec * rhs1 * rhs_scale_vec;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    sum += static_cast<float>(lhs[index]) * lhs_scale * static_cast<float>(rhs[index]) * rhs_scale;
  }
  return sum;
}

float bf16_l2_scale(const BFloat16* values, int64_t size, float post_scale) {
  return 1.0f / std::sqrt(bf16_dot_scaled(values, values, size, 1.0f, 1.0f) + 1.0e-6f) * post_scale;
}

float bf16_l2_scale_and_cache(const BFloat16* values, float* cache, int64_t size, float post_scale) {
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  constexpr int64_t kFloatLanes = FloatVec::size();
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    FloatVec value0;
    FloatVec value1;
    std::tie(value0, value1) = at::vec::convert_to_float(BFloatVec::loadu(values + index));
    value0.store(cache + index);
    value1.store(cache + index + kFloatLanes);
    sum0 = sum0 + value0 * value0;
    sum1 = sum1 + value1 * value1;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    const float value = static_cast<float>(values[index]);
    cache[index] = value;
    sum += value * value;
  }
  return 1.0f / std::sqrt(sum + 1.0e-6f) * post_scale;
}

void bf16_to_fp32(const BFloat16* values, float* output, int64_t size) {
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  constexpr int64_t kFloatLanes = FloatVec::size();
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    FloatVec value0;
    FloatVec value1;
    std::tie(value0, value1) = at::vec::convert_to_float(BFloatVec::loadu(values + index));
    value0.store(output + index);
    value1.store(output + index + kFloatLanes);
  }
  for (; index < size; ++index) {
    output[index] = static_cast<float>(values[index]);
  }
}

float state_dot(
    const float* state, const BFloat16* vector, int64_t size, float state_scale, float vector_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  static_assert(kBFloatLanes == 2 * kFloatLanes);
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec state_scale_vec(state_scale);
  const FloatVec vector_scale_vec(vector_scale);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    const FloatVec state0 = FloatVec::loadu(state + index);
    const FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes);
    const BFloatVec vector_bf16 = BFloatVec::loadu(vector + index);
    FloatVec vector0;
    FloatVec vector1;
    std::tie(vector0, vector1) = at::vec::convert_to_float(vector_bf16);
    sum0 = sum0 + state0 * state_scale_vec * vector0 * vector_scale_vec;
    sum1 = sum1 + state1 * state_scale_vec * vector1 * vector_scale_vec;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    sum += state[index] * state_scale * static_cast<float>(vector[index]) * vector_scale;
  }
  return sum;
}

// The recurrent rule uses the decayed state twice: once for the key
// prediction and once as the base of the rank-one update.  The original path
// recomputes state * decay in both passes.  Store the first result so the
// update pass can reuse it.  Unlike factoring scales out of the reduction,
// this preserves the original per-lane floating-point expression exactly.
float decay_state_and_dot(
    float* state, const BFloat16* vector, int64_t size, float decay, float vector_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  static_assert(kBFloatLanes == 2 * kFloatLanes);
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec decay_vec(decay);
  const FloatVec vector_scale_vec(vector_scale);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    FloatVec state0 = FloatVec::loadu(state + index) * decay_vec;
    FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes) * decay_vec;
    state0.store(state + index);
    state1.store(state + index + kFloatLanes);
    FloatVec vector0;
    FloatVec vector1;
    std::tie(vector0, vector1) = at::vec::convert_to_float(BFloatVec::loadu(vector + index));
    sum0 = sum0 + state0 * vector0 * vector_scale_vec;
    sum1 = sum1 + state1 * vector1 * vector_scale_vec;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    const float decayed = state[index] * decay;
    state[index] = decayed;
    sum += decayed * static_cast<float>(vector[index]) * vector_scale;
  }
  return sum;
}

float state_dot_factored(
    const float* state, const BFloat16* vector, int64_t size, float state_scale, float vector_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  static_assert(kBFloatLanes == 2 * kFloatLanes);
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    const FloatVec state0 = FloatVec::loadu(state + index);
    const FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes);
    FloatVec vector0;
    FloatVec vector1;
    std::tie(vector0, vector1) = at::vec::convert_to_float(BFloatVec::loadu(vector + index));
    sum0 = sum0 + state0 * vector0;
    sum1 = sum1 + state1 * vector1;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    sum += state[index] * static_cast<float>(vector[index]);
  }
  return sum * (state_scale * vector_scale);
}

float update_state_and_dot(float* state,
                           const BFloat16* key,
                           const BFloat16* query,
                           int64_t size,
                           float decay,
                           float delta,
                           float key_scale,
                           float query_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  static_assert(kBFloatLanes == 2 * kFloatLanes);
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec decay_vec(decay);
  const FloatVec delta_vec(delta);
  const FloatVec key_scale_vec(key_scale);
  const FloatVec query_scale_vec(query_scale);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    FloatVec state0 = FloatVec::loadu(state + index);
    FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes);
    const BFloatVec key_bf16 = BFloatVec::loadu(key + index);
    const BFloatVec query_bf16 = BFloatVec::loadu(query + index);
    FloatVec key0;
    FloatVec key1;
    FloatVec query0;
    FloatVec query1;
    std::tie(key0, key1) = at::vec::convert_to_float(key_bf16);
    std::tie(query0, query1) = at::vec::convert_to_float(query_bf16);
    state0 = state0 * decay_vec + key0 * key_scale_vec * delta_vec;
    state1 = state1 * decay_vec + key1 * key_scale_vec * delta_vec;
    state0.store(state + index);
    state1.store(state + index + kFloatLanes);
    sum0 = sum0 + state0 * query0 * query_scale_vec;
    sum1 = sum1 + state1 * query1 * query_scale_vec;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    const float updated = state[index] * decay + static_cast<float>(key[index]) * key_scale * delta;
    state[index] = updated;
    sum += updated * static_cast<float>(query[index]) * query_scale;
  }
  return sum;
}

float update_decayed_state_and_dot(float* state,
                                   const BFloat16* key,
                                   const BFloat16* query,
                                   int64_t size,
                                   float delta,
                                   float key_scale,
                                   float query_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  static_assert(kBFloatLanes == 2 * kFloatLanes);
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec delta_vec(delta);
  const FloatVec key_scale_vec(key_scale);
  const FloatVec query_scale_vec(query_scale);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    FloatVec state0 = FloatVec::loadu(state + index);
    FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes);
    FloatVec key0;
    FloatVec key1;
    FloatVec query0;
    FloatVec query1;
    std::tie(key0, key1) = at::vec::convert_to_float(BFloatVec::loadu(key + index));
    std::tie(query0, query1) = at::vec::convert_to_float(BFloatVec::loadu(query + index));
    state0 = state0 + key0 * key_scale_vec * delta_vec;
    state1 = state1 + key1 * key_scale_vec * delta_vec;
    state0.store(state + index);
    state1.store(state + index + kFloatLanes);
    sum0 = sum0 + state0 * query0 * query_scale_vec;
    sum1 = sum1 + state1 * query1 * query_scale_vec;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    const float updated = state[index] + static_cast<float>(key[index]) * key_scale * delta;
    state[index] = updated;
    sum += updated * static_cast<float>(query[index]) * query_scale;
  }
  return sum;
}

float update_state_and_dot_factored(float* state,
                                    const BFloat16* key,
                                    const BFloat16* query,
                                    int64_t size,
                                    float decay,
                                    float delta,
                                    float key_scale,
                                    float query_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  static_assert(kBFloatLanes == 2 * kFloatLanes);
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec decay_vec(decay);
  const FloatVec key_delta_vec(key_scale * delta);
  int64_t index = 0;
  for (; index + kBFloatLanes <= size; index += kBFloatLanes) {
    FloatVec state0 = FloatVec::loadu(state + index);
    FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes);
    FloatVec key0;
    FloatVec key1;
    FloatVec query0;
    FloatVec query1;
    std::tie(key0, key1) = at::vec::convert_to_float(BFloatVec::loadu(key + index));
    std::tie(query0, query1) = at::vec::convert_to_float(BFloatVec::loadu(query + index));
    state0 = state0 * decay_vec + key0 * key_delta_vec;
    state1 = state1 * decay_vec + key1 * key_delta_vec;
    state0.store(state + index);
    state1.store(state + index + kFloatLanes);
    sum0 = sum0 + state0 * query0;
    sum1 = sum1 + state1 * query1;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    const float updated = state[index] * decay + static_cast<float>(key[index]) * key_scale * delta;
    state[index] = updated;
    sum += updated * static_cast<float>(query[index]);
  }
  return sum * query_scale;
}

float state_dot_fp32_vector(
    const float* state, const float* vector, int64_t size, float state_scale, float vector_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec state_scale_vec(state_scale);
  const FloatVec vector_scale_vec(vector_scale);
  int64_t index = 0;
  for (; index + 2 * kFloatLanes <= size; index += 2 * kFloatLanes) {
    const FloatVec state0 = FloatVec::loadu(state + index);
    const FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes);
    const FloatVec vector0 = FloatVec::loadu(vector + index);
    const FloatVec vector1 = FloatVec::loadu(vector + index + kFloatLanes);
    sum0 = sum0 + state0 * state_scale_vec * vector0 * vector_scale_vec;
    sum1 = sum1 + state1 * state_scale_vec * vector1 * vector_scale_vec;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    sum += state[index] * state_scale * vector[index] * vector_scale;
  }
  return sum;
}

float update_state_and_dot_fp32_vectors(float* state,
                                        const float* key,
                                        const float* query,
                                        int64_t size,
                                        float decay,
                                        float delta,
                                        float key_scale,
                                        float query_scale) {
  constexpr int64_t kFloatLanes = FloatVec::size();
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  const FloatVec decay_vec(decay);
  const FloatVec delta_vec(delta);
  const FloatVec key_scale_vec(key_scale);
  const FloatVec query_scale_vec(query_scale);
  int64_t index = 0;
  for (; index + 2 * kFloatLanes <= size; index += 2 * kFloatLanes) {
    FloatVec state0 = FloatVec::loadu(state + index);
    FloatVec state1 = FloatVec::loadu(state + index + kFloatLanes);
    const FloatVec key0 = FloatVec::loadu(key + index);
    const FloatVec key1 = FloatVec::loadu(key + index + kFloatLanes);
    const FloatVec query0 = FloatVec::loadu(query + index);
    const FloatVec query1 = FloatVec::loadu(query + index + kFloatLanes);
    state0 = state0 * decay_vec + key0 * key_scale_vec * delta_vec;
    state1 = state1 * decay_vec + key1 * key_scale_vec * delta_vec;
    state0.store(state + index);
    state1.store(state + index + kFloatLanes);
    sum0 = sum0 + state0 * query0 * query_scale_vec;
    sum1 = sum1 + state1 * query1 * query_scale_vec;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& a, const FloatVec& b) { return a + b; }, sum0 + sum1);
  for (; index < size; ++index) {
    const float updated = state[index] * decay + key[index] * key_scale * delta;
    state[index] = updated;
    sum += updated * query[index] * query_scale;
  }
  return sum;
}

void gdn_rmsnorm_gated_row(BFloat16* output,
                           const BFloat16* input,
                           const BFloat16* weight,
                           const BFloat16* gate,
                           int64_t hidden,
                           double epsilon,
                           bool sigmoid_gate) {
  constexpr int64_t kBFloatLanes = BFloatVec::size();
  FloatVec sum0(0.0f);
  FloatVec sum1(0.0f);
  int64_t index = 0;
  for (; index + kBFloatLanes <= hidden; index += kBFloatLanes) {
    FloatVec x0;
    FloatVec x1;
    std::tie(x0, x1) = at::vec::convert_to_float(BFloatVec::loadu(input + index));
    sum0 = sum0 + x0 * x0;
    sum1 = sum1 + x1 * x1;
  }
  float sum =
      at::vec::vec_reduce_all<float>([](const FloatVec& lhs, const FloatVec& rhs) { return lhs + rhs; },
                                     sum0 + sum1);
  for (; index < hidden; ++index) {
    const float value = static_cast<float>(input[index]);
    sum += value * value;
  }
  const float inv_rms_value = 1.0f / std::sqrt(sum / static_cast<float>(hidden) + epsilon);
  const FloatVec inv_rms(inv_rms_value);
  const FloatVec one(1.0f);
  index = 0;
  for (; index + kBFloatLanes <= hidden; index += kBFloatLanes) {
    FloatVec x0;
    FloatVec x1;
    FloatVec z0;
    FloatVec z1;
    FloatVec w0;
    FloatVec w1;
    std::tie(x0, x1) = at::vec::convert_to_float(BFloatVec::loadu(input + index));
    std::tie(z0, z1) = at::vec::convert_to_float(BFloatVec::loadu(gate + index));
    std::tie(w0, w1) = at::vec::convert_to_float(BFloatVec::loadu(weight + index));
    const FloatVec gate0 = sigmoid_gate ? one / (one + (-z0).exp()) : z0 / (one + (-z0).exp());
    const FloatVec gate1 = sigmoid_gate ? one / (one + (-z1).exp()) : z1 / (one + (-z1).exp());
    const BFloatVec converted =
        at::vec::convert_from_float<BFloat16>(x0 * inv_rms * w0 * gate0, x1 * inv_rms * w1 * gate1);
    converted.store(output + index);
  }
  for (; index < hidden; ++index) {
    const float gate_value = static_cast<float>(gate[index]);
    const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
    const float activation = sigmoid_gate ? sigmoid : gate_value * sigmoid;
    output[index] = BFloat16(static_cast<float>(input[index]) * inv_rms_value *
                             static_cast<float>(weight[index]) * activation);
  }
}

at::Tensor gdn_decode_impl(const at::Tensor& A_log,
                           const at::Tensor& a,
                           const at::Tensor& b,
                           const at::Tensor& dt_bias,
                           const at::Tensor& query,
                           const at::Tensor& key,
                           const at::Tensor& value,
                           at::Tensor& initial_state,
                           const at::Tensor& state_indices,
                           bool use_qk_l2norm_in_kernel,
                           at::Tensor* output_override,
                           const at::Tensor* norm_weight,
                           const at::Tensor* norm_gate,
                           double norm_epsilon,
                           bool sigmoid_gate) {
  TORCH_CHECK(query.device().is_cpu() && key.device().is_cpu() && value.device().is_cpu() &&
                  initial_state.device().is_cpu(),
              "GDN decode expects CPU tensors");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
                  value.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kBFloat16 &&
                  b.scalar_type() == at::kBFloat16 && dt_bias.scalar_type() == at::kBFloat16,
              "GDN decode expects BF16 activations and gate parameters");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && initial_state.scalar_type() == at::kFloat,
              "GDN decode expects FP32 A_log and recurrent state");
  TORCH_CHECK(query.dim() == 4 && key.sizes() == query.sizes(),
              "query and key must have shape [1, tokens, key_heads, key_dim]");
  TORCH_CHECK(query.size(0) == 1 && value.dim() == 4 && value.size(0) == 1 && value.size(1) == query.size(1),
              "GDN decode supports the vLLM [1, tokens, heads, dim] layout");

  const int64_t tokens = query.size(1);
  const int64_t key_heads = query.size(2);
  const int64_t key_dim = query.size(3);
  const int64_t value_heads = value.size(2);
  const int64_t value_dim = value.size(3);
  TORCH_CHECK(value_heads % key_heads == 0, "value heads must be divisible by key heads");
  TORCH_CHECK(a.sizes() == at::IntArrayRef({tokens, value_heads}) && b.sizes() == a.sizes() &&
                  A_log.numel() == value_heads && dt_bias.numel() == value_heads,
              "invalid GDN gate shapes");
  TORCH_CHECK(initial_state.dim() == 4 && initial_state.size(1) == value_heads &&
                  initial_state.size(2) == value_dim && initial_state.size(3) == key_dim,
              "state must have shape [slots, value_heads, value_dim, key_dim]");
  TORCH_CHECK(initial_state.stride(3) == 1 && initial_state.stride(2) == key_dim &&
                  initial_state.stride(1) == value_dim * key_dim,
              "GDN state must be dense within each cache slot");
  TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.numel() == tokens && state_indices.scalar_type() == at::kInt,
      "state_indices must be INT32 with one entry per token");
  TORCH_CHECK(query.stride(3) == 1 && key.stride(3) == 1 && value.stride(3) == 1,
              "GDN inputs must be contiguous in their head dimension");
  TORCH_CHECK((norm_weight == nullptr) == (norm_gate == nullptr),
              "GDN fused norm requires both weight and gate");
  if (norm_weight != nullptr) {
    TORCH_CHECK(norm_weight->scalar_type() == at::kBFloat16 && norm_weight->is_contiguous() &&
                    norm_weight->dim() == 1 && norm_weight->numel() == value_dim &&
                    norm_gate->scalar_type() == at::kBFloat16 && norm_gate->is_contiguous() &&
                    norm_gate->sizes() == at::IntArrayRef({tokens, value_heads, value_dim}),
                "GDN fused norm expects BF16 gate [tokens,H,V] and weight [V]");
  }

  ScopedGdnThreads scoped_threads("FLAGGEMS_GDN_DECODE_THREADS");
  const bool cache_qk = env_flag("FLAGGEMS_GDN_DECODE_FP32_QK_CACHE", false);
  std::vector<float> query_cache(cache_qk ? tokens * key_heads * key_dim : 0);
  std::vector<float> key_cache(query_cache.size());
  const int64_t head_group = value_heads / key_heads;
  const float query_post_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  std::vector<float> query_scales(tokens * key_heads, query_post_scale);
  std::vector<float> key_scales(tokens * key_heads, 1.0f);
  std::vector<float> decays(tokens * value_heads);
  std::vector<float> betas(tokens * value_heads);

  const auto* query_ptr = query.data_ptr<BFloat16>();
  const auto* key_ptr = key.data_ptr<BFloat16>();
  const auto* value_ptr = value.data_ptr<BFloat16>();
  const auto* a_ptr = a.data_ptr<BFloat16>();
  const auto* b_ptr = b.data_ptr<BFloat16>();
  const auto* A_log_ptr = A_log.data_ptr<float>();
  const auto* dt_bias_ptr = dt_bias.data_ptr<BFloat16>();
  if (use_qk_l2norm_in_kernel) {
    for (int64_t token = 0; token < tokens; ++token) {
      for (int64_t head = 0; head < key_heads; ++head) {
        const int64_t scale_index = token * key_heads + head;
        const auto* q_head = query_ptr + token * query.stride(1) + head * query.stride(2);
        const auto* k_head = key_ptr + token * key.stride(1) + head * key.stride(2);
        if (cache_qk) {
          query_scales[scale_index] = bf16_l2_scale_and_cache(q_head,
                                                              query_cache.data() + scale_index * key_dim,
                                                              key_dim,
                                                              query_post_scale);
          key_scales[scale_index] =
              bf16_l2_scale_and_cache(k_head, key_cache.data() + scale_index * key_dim, key_dim, 1.0f);
        } else {
          query_scales[scale_index] = bf16_l2_scale(q_head, key_dim, query_post_scale);
          key_scales[scale_index] = bf16_l2_scale(k_head, key_dim, 1.0f);
        }
      }
    }
  }
  if (cache_qk && !use_qk_l2norm_in_kernel) {
    for (int64_t token = 0; token < tokens; ++token) {
      for (int64_t head = 0; head < key_heads; ++head) {
        const int64_t offset = (token * key_heads + head) * key_dim;
        bf16_to_fp32(query_ptr + token * query.stride(1) + head * query.stride(2),
                     query_cache.data() + offset,
                     key_dim);
        bf16_to_fp32(key_ptr + token * key.stride(1) + head * key.stride(2),
                     key_cache.data() + offset,
                     key_dim);
      }
    }
  }
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t head = 0; head < value_heads; ++head) {
      const int64_t index = token * value_heads + head;
      const float gate =
          -std::exp(A_log_ptr[head]) *
          softplus(static_cast<float>(a_ptr[index]) + static_cast<float>(dt_bias_ptr[head]), 20.0);
      decays[index] = std::exp(gate);
      betas[index] = 1.0f / (1.0f + std::exp(-static_cast<float>(b_ptr[index])));
    }
  }

  at::Tensor output = output_override == nullptr ? at::empty_like(value) : *output_override;
  TORCH_CHECK(
      output.scalar_type() == at::kBFloat16 && output.sizes() == value.sizes() && output.is_contiguous(),
      "GDN decode output must be contiguous BF16 with value shape");
  auto* output_ptr = output.data_ptr<BFloat16>();
  auto* state_ptr = initial_state.data_ptr<float>();
  const auto* state_index_ptr = state_indices.data_ptr<int32_t>();
  const auto* norm_weight_ptr = norm_weight == nullptr ? nullptr : norm_weight->data_ptr<BFloat16>();
  const auto* norm_gate_ptr = norm_gate == nullptr ? nullptr : norm_gate->data_ptr<BFloat16>();
  at::parallel_for(0, tokens * value_heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t work = begin; work < end; ++work) {
      const int64_t token = work / value_heads;
      const int64_t value_head = work % value_heads;
      const int64_t key_head = value_head / head_group;
      const int32_t slot = state_index_ptr[token];
      TORCH_CHECK(slot >= 0 && slot < initial_state.size(0), "GDN state index out of range");
      const int64_t scale_index = token * key_heads + key_head;
      const float query_scale = query_scales[scale_index];
      const float key_scale = key_scales[scale_index];
      const float decay = decays[work];
      const float beta = betas[work];
      const auto* q_head = query_ptr + token * query.stride(1) + key_head * query.stride(2);
      const auto* k_head = key_ptr + token * key.stride(1) + key_head * key.stride(2);
      const auto* v_head = value_ptr + token * value.stride(1) + value_head * value.stride(2);
      auto* out_head = output_ptr + token * output.stride(1) + value_head * output.stride(2);
      float* state_head = state_ptr + slot * initial_state.stride(0) + value_head * initial_state.stride(1);
      for (int64_t value_index = 0; value_index < value_dim; ++value_index) {
        float* state_row = state_head + value_index * initial_state.stride(2);
        const float predicted = cache_qk ? state_dot_fp32_vector(state_row,
                                                                 key_cache.data() + scale_index * key_dim,
                                                                 key_dim,
                                                                 decay,
                                                                 key_scale)
                                         : state_dot(state_row, k_head, key_dim, decay, key_scale);
        const float delta = (static_cast<float>(v_head[value_index]) - predicted) * beta;
        out_head[value_index] =
            BFloat16(cache_qk ? update_state_and_dot_fp32_vectors(state_row,
                                                                  key_cache.data() + scale_index * key_dim,
                                                                  query_cache.data() + scale_index * key_dim,
                                                                  key_dim,
                                                                  decay,
                                                                  delta,
                                                                  key_scale,
                                                                  query_scale)
                              : update_state_and_dot(state_row,
                                                     k_head,
                                                     q_head,
                                                     key_dim,
                                                     decay,
                                                     delta,
                                                     key_scale,
                                                     query_scale));
      }
      if (norm_weight_ptr != nullptr) {
        const auto* gate_head =
            norm_gate_ptr + token * norm_gate->stride(0) + value_head * norm_gate->stride(1);
        gdn_rmsnorm_gated_row(out_head,
                              out_head,
                              norm_weight_ptr,
                              gate_head,
                              value_dim,
                              norm_epsilon,
                              sigmoid_gate);
      }
    }
  });
  return output;
}

at::Tensor gdn_decode_cpu(const at::Tensor& A_log,
                          const at::Tensor& a,
                          const at::Tensor& b,
                          const at::Tensor& dt_bias,
                          const at::Tensor& query,
                          const at::Tensor& key,
                          const at::Tensor& value,
                          at::Tensor& initial_state,
                          const at::Tensor& state_indices,
                          bool use_qk_l2norm_in_kernel) {
  return gdn_decode_impl(A_log,
                         a,
                         b,
                         dt_bias,
                         query,
                         key,
                         value,
                         initial_state,
                         state_indices,
                         use_qk_l2norm_in_kernel,
                         nullptr,
                         nullptr,
                         nullptr,
                         0.0,
                         false);
}

__attribute__((noinline)) at::Tensor gdn_conv1d_update_cpu(const at::Tensor& input,
                                                           at::Tensor& conv_state,
                                                           const at::Tensor& weight,
                                                           const c10::optional<at::Tensor>& bias,
                                                           const at::Tensor& state_indices,
                                                           bool silu_activation) {
  TORCH_CHECK(input.device().is_cpu() && conv_state.device().is_cpu() && weight.device().is_cpu(),
              "GDN conv update expects CPU tensors");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && conv_state.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "GDN conv update expects BF16 tensors");
  TORCH_CHECK(input.dim() == 2 && conv_state.dim() == 3 && weight.dim() == 2,
              "GDN conv update expects input [tokens, dim], state "
              "[slots, dim, width-1], and weight [dim, width]");
  const int64_t tokens = input.size(0);
  const int64_t dim = input.size(1);
  const int64_t width = weight.size(1);
  TORCH_CHECK(
      width >= 1 && conv_state.size(1) == dim && conv_state.size(2) == width - 1 && weight.size(0) == dim,
      "GDN conv update shape mismatch");
  TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.numel() == tokens && state_indices.scalar_type() == at::kInt,
      "GDN conv state indices must be INT32 with one entry per token");
  TORCH_CHECK(
      !bias.has_value() || (bias->scalar_type() == at::kBFloat16 && bias->dim() == 1 && bias->numel() == dim),
      "GDN conv bias must be BF16 [dim]");

  const auto* input_ptr = input.data_ptr<BFloat16>();
  auto* state_ptr = conv_state.data_ptr<BFloat16>();
  const auto* weight_ptr = weight.data_ptr<BFloat16>();
  const auto* bias_ptr = bias.has_value() ? bias->data_ptr<BFloat16>() : nullptr;
  const auto* index_ptr = state_indices.data_ptr<int32_t>();
  at::Tensor output = at::empty_like(input);
  auto* output_ptr = output.data_ptr<BFloat16>();

  at::parallel_for(0, tokens * dim, 1, [&](int64_t begin, int64_t end) {
    for (int64_t work = begin; work < end; ++work) {
      const int64_t token = work / dim;
      const int64_t channel = work % dim;
      const int32_t slot = index_ptr[token];
      TORCH_CHECK(slot >= 0 && slot < conv_state.size(0), "GDN conv state index out of range");
      const auto* weight_row = weight_ptr + channel * weight.stride(0);
      auto* state_row = state_ptr + slot * conv_state.stride(0) + channel * conv_state.stride(1);
      const float input_value = static_cast<float>(input_ptr[token * input.stride(0) + channel]);
      float value = bias_ptr == nullptr ? 0.0f : static_cast<float>(bias_ptr[channel]);
      for (int64_t index = 0; index + 1 < width; ++index) {
        value += static_cast<float>(state_row[index * conv_state.stride(2)]) *
                 static_cast<float>(weight_row[index * weight.stride(1)]);
      }
      value += input_value * static_cast<float>(weight_row[(width - 1) * weight.stride(1)]);
      for (int64_t index = 0; index + 2 < width; ++index) {
        state_row[index * conv_state.stride(2)] = state_row[(index + 1) * conv_state.stride(2)];
      }
      if (width > 1) {
        state_row[(width - 2) * conv_state.stride(2)] = BFloat16(input_value);
      }
      if (silu_activation) {
        value = value / (1.0f + std::exp(-value));
      }
      output_ptr[token * output.stride(0) + channel] = BFloat16(value);
    }
  });
  return output;
}

void gdn_packed_decode_impl(const at::Tensor& mixed_qkv,
                            const at::Tensor& a,
                            const at::Tensor& b,
                            const at::Tensor& A_log,
                            const at::Tensor& dt_bias,
                            at::Tensor& conv_state,
                            const at::Tensor& conv_weight,
                            const c10::optional<at::Tensor>& conv_bias,
                            at::Tensor& recurrent_state,
                            const at::Tensor& state_indices,
                            at::Tensor& output,
                            bool use_qk_l2norm_in_kernel,
                            const at::Tensor* norm_gate,
                            const at::Tensor* norm_weight,
                            double norm_epsilon,
                            bool sigmoid_gate) {
  ScopedGdnThreads scoped_threads("FLAGGEMS_GDN_DECODE_THREADS");
  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.is_contiguous(),
              "packed GDN decode expects contiguous [tokens, qkv_dim]");
  TORCH_CHECK(recurrent_state.dim() == 4 && output.dim() == 3,
              "packed GDN decode expects recurrent [slots,H,V,K] and "
              "output [tokens,H,V]");
  const int64_t tokens = mixed_qkv.size(0);
  const int64_t value_heads = recurrent_state.size(1);
  const int64_t value_dim = recurrent_state.size(2);
  const int64_t key_dim = recurrent_state.size(3);
  const int64_t value_width = value_heads * value_dim;
  const int64_t qk_width = mixed_qkv.size(1) - value_width;
  TORCH_CHECK(qk_width > 0 && qk_width % (2 * key_dim) == 0,
              "packed GDN decode cannot infer key heads from qkv_dim");
  const int64_t key_heads = qk_width / (2 * key_dim);
  TORCH_CHECK(
      value_heads % key_heads == 0 && output.sizes() == at::IntArrayRef({tokens, value_heads, value_dim}),
      "packed GDN decode head/output shape mismatch");

#if defined(_OPENMP)
  const bool persistent_decode =
      env_flag("FLAGGEMS_GDN_NATIVE_PERSISTENT_DECODE", false) && tokens == 1 && norm_gate != nullptr &&
      norm_weight != nullptr && mixed_qkv.device().is_cpu() && conv_state.device().is_cpu() &&
      conv_weight.device().is_cpu() && recurrent_state.device().is_cpu() && output.device().is_cpu() &&
      a.device().is_cpu() && b.device().is_cpu() && A_log.device().is_cpu() && dt_bias.device().is_cpu() &&
      state_indices.device().is_cpu() && norm_gate->device().is_cpu() && norm_weight->device().is_cpu() &&
      mixed_qkv.scalar_type() == at::kBFloat16 && conv_state.scalar_type() == at::kBFloat16 &&
      conv_weight.scalar_type() == at::kBFloat16 && recurrent_state.scalar_type() == at::kFloat &&
      output.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kBFloat16 &&
      b.scalar_type() == at::kBFloat16 && A_log.scalar_type() == at::kFloat &&
      dt_bias.scalar_type() == at::kBFloat16 && state_indices.scalar_type() == at::kInt &&
      state_indices.dim() == 1 && state_indices.numel() == 1 && state_indices.is_contiguous() &&
      a.is_contiguous() && b.is_contiguous() && A_log.is_contiguous() && dt_bias.is_contiguous() &&
      a.sizes() == at::IntArrayRef({1, value_heads}) && b.sizes() == a.sizes() &&
      A_log.numel() == value_heads && dt_bias.numel() == value_heads && conv_state.dim() == 3 &&
      conv_state.size(1) == mixed_qkv.size(1) && conv_state.size(2) == 3 && conv_weight.dim() == 2 &&
      conv_weight.size(0) == mixed_qkv.size(1) && conv_weight.size(1) == 4 &&
      (!conv_bias.has_value() ||
       (conv_bias->device().is_cpu() && conv_bias->scalar_type() == at::kBFloat16 && conv_bias->dim() == 1 &&
        conv_bias->numel() == mixed_qkv.size(1) && conv_bias->is_contiguous())) &&
      output.is_contiguous() && norm_gate->scalar_type() == at::kBFloat16 && norm_gate->is_contiguous() &&
      norm_gate->sizes() == output.sizes() && norm_weight->scalar_type() == at::kBFloat16 &&
      norm_weight->is_contiguous() && norm_weight->dim() == 1 && norm_weight->numel() == value_dim &&
      recurrent_state.stride(3) == 1 && recurrent_state.stride(2) == key_dim &&
      recurrent_state.stride(1) == value_dim * key_dim;
  if (persistent_decode) {
    const int32_t slot = state_indices.data_ptr<int32_t>()[0];
    TORCH_CHECK(slot >= 0 && slot < recurrent_state.size(0) && slot < conv_state.size(0),
                "GDN state index out of range");
    at::Tensor conv_output = at::empty_like(mixed_qkv);
    std::vector<float> query_scales(key_heads, 1.0f / std::sqrt(static_cast<float>(key_dim)));
    std::vector<float> key_scales(key_heads, 1.0f);
    std::vector<float> decays(value_heads);
    std::vector<float> betas(value_heads);
    const bool fp32_qk_cache = env_flag("FLAGGEMS_GDN_DECODE_FP32_QK_CACHE", false);
    std::vector<float> query_cache(fp32_qk_cache ? key_heads * key_dim : 0);
    std::vector<float> key_cache(fp32_qk_cache ? key_heads * key_dim : 0);

    const auto* input_ptr = mixed_qkv.data_ptr<BFloat16>();
    auto* conv_state_ptr = conv_state.data_ptr<BFloat16>();
    const auto* conv_weight_ptr = conv_weight.data_ptr<BFloat16>();
    const auto* conv_bias_ptr = conv_bias.has_value() ? conv_bias->data_ptr<BFloat16>() : nullptr;
    auto* conv_output_ptr = conv_output.data_ptr<BFloat16>();
    const auto* a_ptr = a.data_ptr<BFloat16>();
    const auto* b_ptr = b.data_ptr<BFloat16>();
    const auto* A_log_ptr = A_log.data_ptr<float>();
    const auto* dt_bias_ptr = dt_bias.data_ptr<BFloat16>();
    auto* recurrent_ptr = recurrent_state.data_ptr<float>();
    auto* output_ptr = output.data_ptr<BFloat16>();
    const auto* norm_gate_ptr = norm_gate->data_ptr<BFloat16>();
    const auto* norm_weight_ptr = norm_weight->data_ptr<BFloat16>();
    const int64_t conv_dim = mixed_qkv.size(1);
    const int64_t key_width = key_heads * key_dim;
    const int64_t head_group = value_heads / key_heads;
    const float query_post_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));

#pragma omp parallel
    {
      const int64_t thread = omp_get_thread_num();
      const int64_t threads = omp_get_num_threads();
      const int64_t channel_begin = conv_dim * thread / threads;
      const int64_t channel_end = conv_dim * (thread + 1) / threads;
      for (int64_t channel = channel_begin; channel < channel_end; ++channel) {
        const auto* weight_row = conv_weight_ptr + channel * conv_weight.stride(0);
        auto* state_row = conv_state_ptr + slot * conv_state.stride(0) + channel * conv_state.stride(1);
        const float input_value = static_cast<float>(input_ptr[channel]);
        float value = conv_bias_ptr == nullptr ? 0.0f : static_cast<float>(conv_bias_ptr[channel]);
        for (int64_t index = 0; index < 3; ++index) {
          value += static_cast<float>(state_row[index * conv_state.stride(2)]) *
                   static_cast<float>(weight_row[index * conv_weight.stride(1)]);
        }
        value += input_value * static_cast<float>(weight_row[3 * conv_weight.stride(1)]);
        state_row[0] = state_row[conv_state.stride(2)];
        state_row[conv_state.stride(2)] = state_row[2 * conv_state.stride(2)];
        state_row[2 * conv_state.stride(2)] = BFloat16(input_value);
        value = value / (1.0f + std::exp(-value));
        conv_output_ptr[channel] = BFloat16(value);
      }

#pragma omp barrier
      const int64_t key_begin = key_heads * thread / threads;
      const int64_t key_end = key_heads * (thread + 1) / threads;
      if (use_qk_l2norm_in_kernel) {
        for (int64_t head = key_begin; head < key_end; ++head) {
          const auto* query_head = conv_output_ptr + head * key_dim;
          const auto* key_head = conv_output_ptr + key_width + head * key_dim;
          query_scales[head] = fp32_qk_cache ? bf16_l2_scale_and_cache(query_head,
                                                                       query_cache.data() + head * key_dim,
                                                                       key_dim,
                                                                       query_post_scale)
                                             : bf16_l2_scale(query_head, key_dim, query_post_scale);
          key_scales[head] =
              fp32_qk_cache
                  ? bf16_l2_scale_and_cache(key_head, key_cache.data() + head * key_dim, key_dim, 1.0f)
                  : bf16_l2_scale(key_head, key_dim, 1.0f);
        }
      } else if (fp32_qk_cache) {
        for (int64_t head = key_begin; head < key_end; ++head) {
          bf16_to_fp32(conv_output_ptr + head * key_dim, query_cache.data() + head * key_dim, key_dim);
          bf16_to_fp32(conv_output_ptr + key_width + head * key_dim,
                       key_cache.data() + head * key_dim,
                       key_dim);
        }
      }
      const int64_t head_begin = value_heads * thread / threads;
      const int64_t head_end = value_heads * (thread + 1) / threads;
      for (int64_t head = head_begin; head < head_end; ++head) {
        const float gate =
            -std::exp(A_log_ptr[head]) *
            softplus(static_cast<float>(a_ptr[head]) + static_cast<float>(dt_bias_ptr[head]), 20.0);
        decays[head] = std::exp(gate);
        betas[head] = 1.0f / (1.0f + std::exp(-static_cast<float>(b_ptr[head])));
      }

#pragma omp barrier
      for (int64_t value_head = head_begin; value_head < head_end; ++value_head) {
        const int64_t key_head = value_head / head_group;
        const auto* q_head = conv_output_ptr + key_head * key_dim;
        const auto* k_head = conv_output_ptr + key_width + key_head * key_dim;
        const auto* v_head = conv_output_ptr + 2 * key_width + value_head * value_dim;
        auto* out_head = output_ptr + value_head * value_dim;
        float* state_head =
            recurrent_ptr + slot * recurrent_state.stride(0) + value_head * recurrent_state.stride(1);
        for (int64_t value_index = 0; value_index < value_dim; ++value_index) {
          float* state_row = state_head + value_index * recurrent_state.stride(2);
          const float predicted =
              fp32_qk_cache ? state_dot_fp32_vector(state_row,
                                                    key_cache.data() + key_head * key_dim,
                                                    key_dim,
                                                    decays[value_head],
                                                    key_scales[key_head])
                            : state_dot(state_row, k_head, key_dim, decays[value_head], key_scales[key_head]);
          const float delta = (static_cast<float>(v_head[value_index]) - predicted) * betas[value_head];
          out_head[value_index] = BFloat16(
              fp32_qk_cache ? update_state_and_dot_fp32_vectors(state_row,
                                                                key_cache.data() + key_head * key_dim,
                                                                query_cache.data() + key_head * key_dim,
                                                                key_dim,
                                                                decays[value_head],
                                                                delta,
                                                                key_scales[key_head],
                                                                query_scales[key_head])
                            : update_state_and_dot(state_row,
                                                   k_head,
                                                   q_head,
                                                   key_dim,
                                                   decays[value_head],
                                                   delta,
                                                   key_scales[key_head],
                                                   query_scales[key_head]));
        }
        gdn_rmsnorm_gated_row(out_head,
                              out_head,
                              norm_weight_ptr,
                              norm_gate_ptr + value_head * value_dim,
                              value_dim,
                              norm_epsilon,
                              sigmoid_gate);
      }
    }
    return;
  }
#endif

  at::Tensor conv_output =
      gdn_conv1d_update_cpu(mixed_qkv, conv_state, conv_weight, conv_bias, state_indices, true);
  const int64_t conv_dim = mixed_qkv.size(1);
  const std::vector<int64_t> qk_sizes = {1, tokens, key_heads, key_dim};
  const std::vector<int64_t> qk_strides = {tokens * conv_dim, conv_dim, key_dim, 1};
  const std::vector<int64_t> value_sizes = {1, tokens, value_heads, value_dim};
  const std::vector<int64_t> value_strides = {tokens * conv_dim, conv_dim, value_dim, 1};
  at::Tensor query = conv_output.as_strided(qk_sizes, qk_strides, 0);
  at::Tensor key = conv_output.as_strided(qk_sizes, qk_strides, key_heads * key_dim);
  at::Tensor value = conv_output.as_strided(value_sizes, value_strides, 2 * key_heads * key_dim);
  if (norm_gate == nullptr && !output.is_contiguous()) {
    at::Tensor recurrent_output = gdn_decode_cpu(A_log,
                                                 a,
                                                 b,
                                                 dt_bias,
                                                 query,
                                                 key,
                                                 value,
                                                 recurrent_state,
                                                 state_indices,
                                                 use_qk_l2norm_in_kernel);
    output.copy_(recurrent_output.view(output.sizes()));
    return;
  }
  // Decode already receives its final output buffer. Write into that buffer
  // directly instead of allocating a recurrent output and copying it again.
  at::Tensor output_view = output.view(value_sizes);
  gdn_decode_impl(A_log,
                  a,
                  b,
                  dt_bias,
                  query,
                  key,
                  value,
                  recurrent_state,
                  state_indices,
                  use_qk_l2norm_in_kernel,
                  &output_view,
                  norm_weight,
                  norm_gate,
                  norm_epsilon,
                  sigmoid_gate);
}

__attribute__((noinline)) void gdn_packed_decode_cpu(const at::Tensor& mixed_qkv,
                                                     const at::Tensor& a,
                                                     const at::Tensor& b,
                                                     const at::Tensor& A_log,
                                                     const at::Tensor& dt_bias,
                                                     at::Tensor& conv_state,
                                                     const at::Tensor& conv_weight,
                                                     const c10::optional<at::Tensor>& conv_bias,
                                                     at::Tensor& recurrent_state,
                                                     const at::Tensor& state_indices,
                                                     at::Tensor& output,
                                                     bool use_qk_l2norm_in_kernel) {
  gdn_packed_decode_impl(mixed_qkv,
                         a,
                         b,
                         A_log,
                         dt_bias,
                         conv_state,
                         conv_weight,
                         conv_bias,
                         recurrent_state,
                         state_indices,
                         output,
                         use_qk_l2norm_in_kernel,
                         nullptr,
                         nullptr,
                         0.0,
                         false);
}

void gdn_packed_decode_norm_cpu_impl(const at::Tensor& mixed_qkv,
                                     const at::Tensor& a,
                                     const at::Tensor& b,
                                     const at::Tensor& A_log,
                                     const at::Tensor& dt_bias,
                                     at::Tensor& conv_state,
                                     const at::Tensor& conv_weight,
                                     const c10::optional<at::Tensor>& conv_bias,
                                     at::Tensor& recurrent_state,
                                     const at::Tensor& state_indices,
                                     const at::Tensor& norm_gate,
                                     const at::Tensor& norm_weight,
                                     double norm_epsilon,
                                     bool sigmoid_gate,
                                     at::Tensor& output,
                                     bool use_qk_l2norm_in_kernel) {
  gdn_packed_decode_impl(mixed_qkv,
                         a,
                         b,
                         A_log,
                         dt_bias,
                         conv_state,
                         conv_weight,
                         conv_bias,
                         recurrent_state,
                         state_indices,
                         output,
                         use_qk_l2norm_in_kernel,
                         &norm_gate,
                         &norm_weight,
                         norm_epsilon,
                         sigmoid_gate);
}

at::Tensor gdn_rmsnorm_gated_cpu(const at::Tensor& input,
                                 const at::Tensor& weight,
                                 const at::Tensor& gate,
                                 double epsilon,
                                 bool sigmoid_gate) {
  TORCH_CHECK(input.device().is_cpu() && weight.device().is_cpu() && gate.device().is_cpu(),
              "GDN gated RMSNorm expects CPU tensors");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
                  gate.scalar_type() == at::kBFloat16,
              "GDN gated RMSNorm expects BF16 tensors");
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous() && weight.is_contiguous() &&
                  input.sizes() == gate.sizes() && weight.dim() == 1 && input.size(-1) == weight.numel(),
              "GDN gated RMSNorm expects matching contiguous [...,hidden] "
              "input/gate and [hidden] weight");
  const int64_t hidden = input.size(-1);
  TORCH_CHECK(hidden > 0, "GDN gated RMSNorm requires positive hidden size");
  const int64_t rows = input.numel() / hidden;
  const auto* input_ptr = input.data_ptr<BFloat16>();
  const auto* gate_ptr = gate.data_ptr<BFloat16>();
  const auto* weight_ptr = weight.data_ptr<BFloat16>();
  at::Tensor output = at::empty_like(input);
  auto* output_ptr = output.data_ptr<BFloat16>();

  const int64_t row_grain = std::max<int64_t>(1, 32768 / hidden);
  at::parallel_for(0, rows, row_grain, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const auto* x = input_ptr + row * hidden;
      const auto* z = gate_ptr + row * hidden;
      auto* out = output_ptr + row * hidden;
      gdn_rmsnorm_gated_row(out, x, weight_ptr, z, hidden, epsilon, sigmoid_gate);
    }
  });
  return output;
}

at::Tensor gdn_prefill_cpu(const at::Tensor& query,
                           const at::Tensor& key,
                           const at::Tensor& value,
                           const at::Tensor& g,
                           const at::Tensor& beta,
                           at::Tensor& initial_state,
                           const at::Tensor& cu_seqlens,
                           bool use_qk_l2norm_in_kernel) {
  ScopedGdnThreads scoped_threads("FLAGGEMS_GDN_PREFILL_THREADS");
  TORCH_CHECK(query.device().is_cpu() && key.device().is_cpu() && value.device().is_cpu() &&
                  g.device().is_cpu() && beta.device().is_cpu() && initial_state.device().is_cpu(),
              "GDN prefill expects CPU tensors");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
                  value.scalar_type() == at::kBFloat16 &&
                  (beta.scalar_type() == at::kBFloat16 || beta.scalar_type() == at::kFloat) &&
                  g.scalar_type() == at::kFloat && initial_state.scalar_type() == at::kFloat,
              "GDN prefill expects BF16 activations, FP32 g, BF16/FP32 beta, "
              "and FP32 state; got query=",
              query.scalar_type(),
              " key=",
              key.scalar_type(),
              " value=",
              value.scalar_type(),
              " g=",
              g.scalar_type(),
              " beta=",
              beta.scalar_type(),
              " state=",
              initial_state.scalar_type());
  TORCH_CHECK(query.dim() == 4 && key.sizes() == query.sizes() && query.size(0) == 1 && value.dim() == 4 &&
                  value.size(0) == 1 && value.size(1) == query.size(1),
              "GDN prefill expects [1, tokens, heads, dim] inputs");

  const int64_t tokens = query.size(1);
  const int64_t key_heads = query.size(2);
  const int64_t key_dim = query.size(3);
  const int64_t value_heads = value.size(2);
  const int64_t value_dim = value.size(3);
  const int64_t sequences = cu_seqlens.numel() - 1;
  TORCH_CHECK(sequences >= 1 && value_heads % key_heads == 0,
              "invalid GDN prefill sequence/head configuration");
  TORCH_CHECK(g.sizes() == at::IntArrayRef({1, tokens, value_heads}) && beta.sizes() == g.sizes(),
              "invalid GDN prefill gate shapes");
  TORCH_CHECK(initial_state.sizes() == at::IntArrayRef({sequences, value_heads, value_dim, key_dim}),
              "invalid GDN prefill state shape");
  TORCH_CHECK(initial_state.stride(3) == 1 && initial_state.stride(2) == key_dim &&
                  initial_state.stride(1) == value_dim * key_dim,
              "GDN prefill state must be dense per sequence");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.dim() == 1,
              "GDN prefill cu_seqlens must be INT32");

  const auto* query_ptr = query.data_ptr<BFloat16>();
  const auto* key_ptr = key.data_ptr<BFloat16>();
  const auto* value_ptr = value.data_ptr<BFloat16>();
  const auto* g_ptr = g.data_ptr<float>();
  const auto* beta_bf16_ptr = beta.scalar_type() == at::kBFloat16 ? beta.data_ptr<BFloat16>() : nullptr;
  const auto* beta_float_ptr = beta.scalar_type() == at::kFloat ? beta.data_ptr<float>() : nullptr;
  const auto* cu_ptr = cu_seqlens.data_ptr<int32_t>();
  auto* state_ptr = initial_state.data_ptr<float>();
  at::Tensor output = at::empty_like(value);
  auto* output_ptr = output.data_ptr<BFloat16>();

  const int64_t head_group = value_heads / key_heads;
  const float query_post_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  std::vector<float> query_scales(tokens * key_heads, query_post_scale);
  std::vector<float> key_scales(tokens * key_heads, 1.0f);
  const char* cache_env = std::getenv("FLAGGEMS_GDN_PREFILL_FP32_QK_CACHE");
  const bool cache_fp32_qk = use_qk_l2norm_in_kernel && cache_env != nullptr &&
                             std::string(cache_env) != "0" && std::string(cache_env) != "false" &&
                             std::string(cache_env) != "off";
  std::vector<float> query_cache;
  std::vector<float> key_cache;
  if (cache_fp32_qk) {
    query_cache.resize(tokens * key_heads * key_dim);
    key_cache.resize(tokens * key_heads * key_dim);
  }
  if (use_qk_l2norm_in_kernel) {
    at::parallel_for(0, tokens * key_heads, 1, [&](int64_t begin, int64_t end) {
      for (int64_t work = begin; work < end; ++work) {
        const int64_t token = work / key_heads;
        const int64_t key_head = work % key_heads;
        const auto* q_head = query_ptr + token * query.stride(1) + key_head * query.stride(2);
        const auto* k_head = key_ptr + token * key.stride(1) + key_head * key.stride(2);
        if (cache_fp32_qk) {
          query_scales[work] =
              bf16_l2_scale_and_cache(q_head, query_cache.data() + work * key_dim, key_dim, query_post_scale);
          key_scales[work] =
              bf16_l2_scale_and_cache(k_head, key_cache.data() + work * key_dim, key_dim, 1.0f);
        } else {
          query_scales[work] = bf16_l2_scale(q_head, key_dim, query_post_scale);
          key_scales[work] = bf16_l2_scale(k_head, key_dim, 1.0f);
        }
      }
    });
  }

  const char* grouped_env = std::getenv("FLAGGEMS_GDN_PREFILL_GROUP_KEY_HEADS");
  const bool group_key_heads = !cache_fp32_qk && grouped_env != nullptr && std::string(grouped_env) != "0" &&
                               std::string(grouped_env) != "false" && std::string(grouped_env) != "off";
  const char* factored_env = std::getenv("FLAGGEMS_GDN_PREFILL_FACTORED_SCALES");
  const bool factored_scales = !cache_fp32_qk && !group_key_heads && factored_env != nullptr &&
                               std::string(factored_env) != "0" && std::string(factored_env) != "false" &&
                               std::string(factored_env) != "off";
  const bool decay_store = !cache_fp32_qk && !group_key_heads && !factored_scales &&
                           env_flag("FLAGGEMS_GDN_PREFILL_DECAY_STORE", true);
  if (group_key_heads) {
    at::parallel_for(0, sequences * key_heads, 1, [&](int64_t begin, int64_t end) {
      std::vector<float> query_token(key_dim);
      std::vector<float> key_token(key_dim);
      for (int64_t work = begin; work < end; ++work) {
        const int64_t sequence = work / key_heads;
        const int64_t key_head = work % key_heads;
        const int64_t token_begin = cu_ptr[sequence];
        const int64_t token_end = cu_ptr[sequence + 1];
        TORCH_CHECK(token_begin >= 0 && token_begin <= token_end && token_end <= tokens,
                    "invalid GDN prefill cu_seqlens");
        for (int64_t token = token_begin; token < token_end; ++token) {
          const int64_t scale_index = token * key_heads + key_head;
          const float query_scale = query_scales[scale_index];
          const float key_scale = key_scales[scale_index];
          const auto* q_head = query_ptr + token * query.stride(1) + key_head * query.stride(2);
          const auto* k_head = key_ptr + token * key.stride(1) + key_head * key.stride(2);
          bf16_to_fp32(q_head, query_token.data(), key_dim);
          bf16_to_fp32(k_head, key_token.data(), key_dim);
          for (int64_t local_head = 0; local_head < head_group; ++local_head) {
            const int64_t value_head = key_head * head_group + local_head;
            float* state_head =
                state_ptr + sequence * initial_state.stride(0) + value_head * initial_state.stride(1);
            const float decay = std::exp(g_ptr[token * g.stride(1) + value_head * g.stride(2)]);
            const int64_t beta_index = token * beta.stride(1) + value_head * beta.stride(2);
            const float beta_value = beta_float_ptr != nullptr
                                         ? beta_float_ptr[beta_index]
                                         : static_cast<float>(beta_bf16_ptr[beta_index]);
            const auto* v_head = value_ptr + token * value.stride(1) + value_head * value.stride(2);
            auto* out_head = output_ptr + token * output.stride(1) + value_head * output.stride(2);
            for (int64_t value_index = 0; value_index < value_dim; ++value_index) {
              float* state_row = state_head + value_index * initial_state.stride(2);
              const float predicted =
                  state_dot_fp32_vector(state_row, key_token.data(), key_dim, decay, key_scale);
              const float delta = (static_cast<float>(v_head[value_index]) - predicted) * beta_value;
              out_head[value_index] = BFloat16(update_state_and_dot_fp32_vectors(state_row,
                                                                                 key_token.data(),
                                                                                 query_token.data(),
                                                                                 key_dim,
                                                                                 decay,
                                                                                 delta,
                                                                                 key_scale,
                                                                                 query_scale));
            }
          }
        }
      }
    });
    return output;
  }

  at::parallel_for(0, sequences * value_heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t work = begin; work < end; ++work) {
      const int64_t sequence = work / value_heads;
      const int64_t value_head = work % value_heads;
      const int64_t key_head = value_head / head_group;
      const int64_t token_begin = cu_ptr[sequence];
      const int64_t token_end = cu_ptr[sequence + 1];
      TORCH_CHECK(token_begin >= 0 && token_begin <= token_end && token_end <= tokens,
                  "invalid GDN prefill cu_seqlens");
      float* state_head =
          state_ptr + sequence * initial_state.stride(0) + value_head * initial_state.stride(1);
      for (int64_t token = token_begin; token < token_end; ++token) {
        const int64_t scale_index = token * key_heads + key_head;
        const float query_scale = query_scales[scale_index];
        const float key_scale = key_scales[scale_index];
        const float decay = std::exp(g_ptr[token * g.stride(1) + value_head * g.stride(2)]);
        const int64_t beta_index = token * beta.stride(1) + value_head * beta.stride(2);
        const float beta_value = beta_float_ptr != nullptr ? beta_float_ptr[beta_index]
                                                           : static_cast<float>(beta_bf16_ptr[beta_index]);
        const auto* q_head = query_ptr + token * query.stride(1) + key_head * query.stride(2);
        const auto* k_head = key_ptr + token * key.stride(1) + key_head * key.stride(2);
        const auto* v_head = value_ptr + token * value.stride(1) + value_head * value.stride(2);
        auto* out_head = output_ptr + token * output.stride(1) + value_head * output.stride(2);
        if (cache_fp32_qk) {
          const auto* q_head_fp32 = query_cache.data() + scale_index * key_dim;
          const auto* k_head_fp32 = key_cache.data() + scale_index * key_dim;
          for (int64_t value_index = 0; value_index < value_dim; ++value_index) {
            float* state_row = state_head + value_index * initial_state.stride(2);
            const float predicted = state_dot_fp32_vector(state_row, k_head_fp32, key_dim, decay, key_scale);
            const float delta = (static_cast<float>(v_head[value_index]) - predicted) * beta_value;
            out_head[value_index] = BFloat16(update_state_and_dot_fp32_vectors(state_row,
                                                                               k_head_fp32,
                                                                               q_head_fp32,
                                                                               key_dim,
                                                                               decay,
                                                                               delta,
                                                                               key_scale,
                                                                               query_scale));
          }
        } else if (factored_scales) {
          for (int64_t value_index = 0; value_index < value_dim; ++value_index) {
            float* state_row = state_head + value_index * initial_state.stride(2);
            const float predicted = state_dot_factored(state_row, k_head, key_dim, decay, key_scale);
            const float delta = (static_cast<float>(v_head[value_index]) - predicted) * beta_value;
            out_head[value_index] = BFloat16(update_state_and_dot_factored(state_row,
                                                                           k_head,
                                                                           q_head,
                                                                           key_dim,
                                                                           decay,
                                                                           delta,
                                                                           key_scale,
                                                                           query_scale));
          }
        } else if (decay_store) {
          for (int64_t value_index = 0; value_index < value_dim; ++value_index) {
            float* state_row = state_head + value_index * initial_state.stride(2);
            const float predicted = decay_state_and_dot(state_row, k_head, key_dim, decay, key_scale);
            const float delta = (static_cast<float>(v_head[value_index]) - predicted) * beta_value;
            out_head[value_index] = BFloat16(update_decayed_state_and_dot(state_row,
                                                                          k_head,
                                                                          q_head,
                                                                          key_dim,
                                                                          delta,
                                                                          key_scale,
                                                                          query_scale));
          }
        } else {
          for (int64_t value_index = 0; value_index < value_dim; ++value_index) {
            float* state_row = state_head + value_index * initial_state.stride(2);
            const float predicted = state_dot(state_row, k_head, key_dim, decay, key_scale);
            const float delta = (static_cast<float>(v_head[value_index]) - predicted) * beta_value;
            out_head[value_index] = BFloat16(update_state_and_dot(state_row,
                                                                  k_head,
                                                                  q_head,
                                                                  key_dim,
                                                                  decay,
                                                                  delta,
                                                                  key_scale,
                                                                  query_scale));
          }
        }
      }
    }
  });
  return output;
}

at::Tensor gdn_conv1d_prefill_cpu(const at::Tensor& input,
                                  at::Tensor& conv_state,
                                  const at::Tensor& weight,
                                  const c10::optional<at::Tensor>& bias,
                                  const at::Tensor& query_start_loc,
                                  const at::Tensor& cache_indices,
                                  const at::Tensor& has_initial_state,
                                  bool silu_activation) {
  TORCH_CHECK(input.device().is_cpu() && conv_state.device().is_cpu() && weight.device().is_cpu(),
              "GDN conv prefill expects CPU tensors");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && conv_state.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "GDN conv prefill expects BF16 tensors");
  TORCH_CHECK(input.dim() == 2 && conv_state.dim() == 3 && weight.dim() == 2,
              "GDN conv prefill expects input [dim, tokens], state "
              "[slots, dim, width-1], and weight [dim, width]");
  const int64_t dim = input.size(0);
  const int64_t tokens = input.size(1);
  const int64_t width = weight.size(1);
  const int64_t sequences = query_start_loc.numel() - 1;
  // Speculative decoding reserves `num_spec` additional history positions in
  // the vLLM convolution cache.  Prefill still consumes and updates the first
  // width-1 causal-history positions; the extra rollback positions are owned
  // by the later speculative update path.  Match vLLM's reference kernel,
  // which accepts state_len >= width-1 instead of requiring exact equality.
  TORCH_CHECK(
      width >= 1 && conv_state.size(1) == dim && conv_state.size(2) >= width - 1 && weight.size(0) == dim,
      "GDN conv prefill shape mismatch");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt && cache_indices.scalar_type() == at::kInt &&
                  cache_indices.numel() == sequences && has_initial_state.numel() == sequences,
              "invalid GDN conv prefill metadata");
  TORCH_CHECK(!bias.has_value() || (bias->scalar_type() == at::kBFloat16 && bias->numel() == dim),
              "invalid GDN conv prefill bias");

  const auto* input_ptr = input.data_ptr<BFloat16>();
  auto* state_ptr = conv_state.data_ptr<BFloat16>();
  const auto* weight_ptr = weight.data_ptr<BFloat16>();
  const auto* bias_ptr = bias.has_value() ? bias->data_ptr<BFloat16>() : nullptr;
  const auto* starts = query_start_loc.data_ptr<int32_t>();
  const auto* slots = cache_indices.data_ptr<int32_t>();
  at::Tensor initial_flags = has_initial_state.to(at::kBool).contiguous();
  const auto* initial = initial_flags.data_ptr<bool>();
  at::Tensor output = at::empty_like(input);
  auto* output_ptr = output.data_ptr<BFloat16>();

  // Validate request metadata once.  The former channel-major loop repeated
  // these checks `dim` times and then walked a token-major transpose with a
  // ~20KB stride for every channel.
  for (int64_t sequence = 0; sequence < sequences; ++sequence) {
    TORCH_CHECK(slots[sequence] >= 0 && slots[sequence] < conv_state.size(0),
                "GDN conv prefill state index out of range");
    TORCH_CHECK(
        starts[sequence] >= 0 && starts[sequence] <= starts[sequence + 1] && starts[sequence + 1] <= tokens,
        "invalid GDN conv prefill query_start_loc");
  }

  const char* layout_env = std::getenv("FLAGGEMS_GDN_CONV_PREFILL_TOKEN_MAJOR");
  const bool token_major_requested = layout_env != nullptr && std::string(layout_env) != "0" &&
                                     std::string(layout_env) != "false" && std::string(layout_env) != "off";
  // vLLM passes transpose([tokens, dim]), whose channel stride is one.  Keep
  // the original loop for genuinely channel-major contiguous callers.
  const bool token_major = token_major_requested && input.stride(0) == 1 && width == 4;
  if (!token_major) {
    at::parallel_for(0, sequences * dim, 1, [&](int64_t begin, int64_t end) {
      for (int64_t work = begin; work < end; ++work) {
        const int64_t sequence = work / dim;
        const int64_t channel = work % dim;
        const int32_t slot = slots[sequence];
        const int64_t token_begin = starts[sequence];
        const int64_t token_end = starts[sequence + 1];
        const auto* weight_row = weight_ptr + channel * weight.stride(0);
        auto* state_row = state_ptr + slot * conv_state.stride(0) + channel * conv_state.stride(1);
        if (!initial[sequence]) {
          for (int64_t index = 0; index + 1 < width; ++index) {
            state_row[index * conv_state.stride(2)] = BFloat16(0.0f);
          }
        }
        for (int64_t token = token_begin; token < token_end; ++token) {
          const float input_value =
              static_cast<float>(input_ptr[channel * input.stride(0) + token * input.stride(1)]);
          float result = bias_ptr == nullptr ? 0.0f : static_cast<float>(bias_ptr[channel]);
          for (int64_t index = 0; index + 1 < width; ++index) {
            result += static_cast<float>(state_row[index * conv_state.stride(2)]) *
                      static_cast<float>(weight_row[index * weight.stride(1)]);
          }
          result += input_value * static_cast<float>(weight_row[(width - 1) * weight.stride(1)]);
          for (int64_t index = 0; index + 2 < width; ++index) {
            state_row[index * conv_state.stride(2)] = state_row[(index + 1) * conv_state.stride(2)];
          }
          if (width > 1) {
            state_row[(width - 2) * conv_state.stride(2)] = BFloat16(input_value);
          }
          if (silu_activation) {
            result = result / (1.0f + std::exp(-result));
          }
          output_ptr[channel * output.stride(0) + token * output.stride(1)] = BFloat16(result);
        }
      }
    });
    return output;
  }

  // The width-four causal convolution is an FIR, not a recurrence: output
  // values never feed later outputs.  Compute directly from the immutable
  // initial history and input, then commit only the final three history values.
  // This experimental layout removes per-token state shifts and makes the
  // inner channel walk contiguous in vLLM's transpose([tokens, dim]) storage.
  constexpr int64_t kChannelBlock = 32;
  const int64_t channel_blocks = (dim + kChannelBlock - 1) / kChannelBlock;
  at::parallel_for(0, sequences * channel_blocks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t work = begin; work < end; ++work) {
      const int64_t sequence = work / channel_blocks;
      const int64_t channel_begin = (work % channel_blocks) * kChannelBlock;
      const int64_t channel_end = std::min(dim, channel_begin + kChannelBlock);
      const int32_t slot = slots[sequence];
      const int64_t token_begin = starts[sequence];
      const int64_t token_end = starts[sequence + 1];
      for (int64_t token = token_begin; token < token_end; ++token) {
        const int64_t relative_token = token - token_begin;
        for (int64_t channel = channel_begin; channel < channel_end; ++channel) {
          const auto* weight_row = weight_ptr + channel * weight.stride(0);
          const auto* state_row = state_ptr + slot * conv_state.stride(0) + channel * conv_state.stride(1);
          const auto input_at = [&](int64_t input_token) {
            return input_ptr[channel * input.stride(0) + input_token * input.stride(1)];
          };
          const BFloat16 zero(0.0f);
          const BFloat16 history0 = initial[sequence] ? state_row[0] : zero;
          const BFloat16 history1 = initial[sequence] ? state_row[conv_state.stride(2)] : zero;
          const BFloat16 history2 = initial[sequence] ? state_row[2 * conv_state.stride(2)] : zero;
          const BFloat16 source0 =
              relative_token >= 3
                  ? input_at(token - 3)
                  : (relative_token == 0 ? history0 : (relative_token == 1 ? history1 : history2));
          const BFloat16 source1 =
              relative_token >= 2 ? input_at(token - 2) : (relative_token == 0 ? history1 : history2);
          const BFloat16 source2 = relative_token >= 1 ? input_at(token - 1) : history2;
          const BFloat16 source3 = input_at(token);
          float result = bias_ptr == nullptr ? 0.0f : static_cast<float>(bias_ptr[channel]);
          result += static_cast<float>(source0) * static_cast<float>(weight_row[0]);
          result += static_cast<float>(source1) * static_cast<float>(weight_row[weight.stride(1)]);
          result += static_cast<float>(source2) * static_cast<float>(weight_row[2 * weight.stride(1)]);
          result += static_cast<float>(source3) * static_cast<float>(weight_row[3 * weight.stride(1)]);
          if (silu_activation) {
            result = result / (1.0f + std::exp(-result));
          }
          output_ptr[channel * output.stride(0) + token * output.stride(1)] = BFloat16(result);
        }
      }
      const int64_t sequence_tokens = token_end - token_begin;
      for (int64_t channel = channel_begin; channel < channel_end; ++channel) {
        auto* state_row = state_ptr + slot * conv_state.stride(0) + channel * conv_state.stride(1);
        const BFloat16 zero(0.0f);
        const BFloat16 old_history0 = initial[sequence] ? state_row[0] : zero;
        const BFloat16 old_history1 = initial[sequence] ? state_row[conv_state.stride(2)] : zero;
        const BFloat16 old_history2 = initial[sequence] ? state_row[2 * conv_state.stride(2)] : zero;
        const auto final_source = [&](int64_t relative_token) {
          if (relative_token >= 0) {
            return input_ptr[channel * input.stride(0) + (token_begin + relative_token) * input.stride(1)];
          }
          if (relative_token == -3) {
            return old_history0;
          }
          if (relative_token == -2) {
            return old_history1;
          }
          return old_history2;
        };
        state_row[0] = final_source(sequence_tokens - 3);
        state_row[conv_state.stride(2)] = final_source(sequence_tokens - 2);
        state_row[2 * conv_state.stride(2)] = final_source(sequence_tokens - 1);
      }
    }
  });
  return output;
}

}  // namespace

void flag_gems::arm::gdn_packed_decode_norm_cpu(const at::Tensor& mixed_qkv,
                                                const at::Tensor& a,
                                                const at::Tensor& b,
                                                const at::Tensor& A_log,
                                                const at::Tensor& dt_bias,
                                                at::Tensor& conv_state,
                                                const at::Tensor& conv_weight,
                                                const c10::optional<at::Tensor>& conv_bias,
                                                at::Tensor& recurrent_state,
                                                const at::Tensor& state_indices,
                                                const at::Tensor& norm_gate,
                                                const at::Tensor& norm_weight,
                                                double norm_epsilon,
                                                bool sigmoid_gate,
                                                at::Tensor& output,
                                                bool use_qk_l2norm_in_kernel) {
  gdn_packed_decode_norm_cpu_impl(mixed_qkv,
                                  a,
                                  b,
                                  A_log,
                                  dt_bias,
                                  conv_state,
                                  conv_weight,
                                  conv_bias,
                                  recurrent_state,
                                  state_indices,
                                  norm_gate,
                                  norm_weight,
                                  norm_epsilon,
                                  sigmoid_gate,
                                  output,
                                  use_qk_l2norm_in_kernel);
}

namespace {
at::Tensor gdn_decode_raw(const at::Tensor& A_log,
                          const at::Tensor& a,
                          const at::Tensor& b,
                          const at::Tensor& dt_bias,
                          const at::Tensor& query,
                          const at::Tensor& key,
                          const at::Tensor& value,
                          at::Tensor& initial_state,
                          const at::Tensor& state_indices,
                          bool use_qk_l2norm_in_kernel) {
  return flag_gems::arm::invoke_raw([&] {
    return gdn_decode_cpu(A_log,
                          a,
                          b,
                          dt_bias,
                          query,
                          key,
                          value,
                          initial_state,
                          state_indices,
                          use_qk_l2norm_in_kernel);
  });
}
at::Tensor gdn_conv1d_update_raw(const at::Tensor& input,
                                 at::Tensor& conv_state,
                                 const at::Tensor& weight,
                                 const c10::optional<at::Tensor>& bias,
                                 const at::Tensor& state_indices,
                                 bool silu_activation) {
  return flag_gems::arm::invoke_raw(
      [&] { return gdn_conv1d_update_cpu(input, conv_state, weight, bias, state_indices, silu_activation); });
}
void gdn_packed_decode_raw(const at::Tensor& mixed_qkv,
                           const at::Tensor& a,
                           const at::Tensor& b,
                           const at::Tensor& A_log,
                           const at::Tensor& dt_bias,
                           at::Tensor& conv_state,
                           const at::Tensor& conv_weight,
                           const c10::optional<at::Tensor>& conv_bias,
                           at::Tensor& recurrent_state,
                           const at::Tensor& state_indices,
                           at::Tensor& output,
                           bool use_qk_l2norm_in_kernel) {
  return flag_gems::arm::invoke_raw([&] {
    return gdn_packed_decode_cpu(mixed_qkv,
                                 a,
                                 b,
                                 A_log,
                                 dt_bias,
                                 conv_state,
                                 conv_weight,
                                 conv_bias,
                                 recurrent_state,
                                 state_indices,
                                 output,
                                 use_qk_l2norm_in_kernel);
  });
}
void gdn_packed_decode_norm_raw(const at::Tensor& mixed_qkv,
                                const at::Tensor& a,
                                const at::Tensor& b,
                                const at::Tensor& A_log,
                                const at::Tensor& dt_bias,
                                at::Tensor& conv_state,
                                const at::Tensor& conv_weight,
                                const c10::optional<at::Tensor>& conv_bias,
                                at::Tensor& recurrent_state,
                                const at::Tensor& state_indices,
                                const at::Tensor& norm_gate,
                                const at::Tensor& norm_weight,
                                double norm_epsilon,
                                bool sigmoid_gate,
                                at::Tensor& output,
                                bool use_qk_l2norm_in_kernel) {
  return flag_gems::arm::invoke_raw([&] {
    return flag_gems::arm::gdn_packed_decode_norm_cpu(mixed_qkv,
                                                      a,
                                                      b,
                                                      A_log,
                                                      dt_bias,
                                                      conv_state,
                                                      conv_weight,
                                                      conv_bias,
                                                      recurrent_state,
                                                      state_indices,
                                                      norm_gate,
                                                      norm_weight,
                                                      norm_epsilon,
                                                      sigmoid_gate,
                                                      output,
                                                      use_qk_l2norm_in_kernel);
  });
}
at::Tensor gdn_rmsnorm_gated_raw(const at::Tensor& input,
                                 const at::Tensor& weight,
                                 const at::Tensor& gate,
                                 double epsilon,
                                 bool sigmoid_gate) {
  return flag_gems::arm::invoke_raw(
      [&] { return gdn_rmsnorm_gated_cpu(input, weight, gate, epsilon, sigmoid_gate); });
}
at::Tensor gdn_prefill_raw(const at::Tensor& query,
                           const at::Tensor& key,
                           const at::Tensor& value,
                           const at::Tensor& g,
                           const at::Tensor& beta,
                           at::Tensor& initial_state,
                           const at::Tensor& cu_seqlens,
                           bool use_qk_l2norm_in_kernel) {
  return flag_gems::arm::invoke_raw([&] {
    return gdn_prefill_cpu(query, key, value, g, beta, initial_state, cu_seqlens, use_qk_l2norm_in_kernel);
  });
}
at::Tensor gdn_conv1d_prefill_raw(const at::Tensor& input,
                                  at::Tensor& conv_state,
                                  const at::Tensor& weight,
                                  const c10::optional<at::Tensor>& bias,
                                  const at::Tensor& query_start_loc,
                                  const at::Tensor& cache_indices,
                                  const at::Tensor& has_initial_state,
                                  bool silu_activation) {
  return flag_gems::arm::invoke_raw([&] {
    return gdn_conv1d_prefill_cpu(input,
                                  conv_state,
                                  weight,
                                  bias,
                                  query_start_loc,
                                  cache_indices,
                                  has_initial_state,
                                  silu_activation);
  });
}
}  // namespace
TORCH_LIBRARY(flag_gems_arm_gdn, m) {
  m.def(
      "gdn_decode(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor query, Tensor key, Tensor value, "
      "Tensor(a!) initial_state, Tensor state_indices, bool use_qk_l2norm_in_kernel=True) -> Tensor");
  m.def(
      "gdn_decode.raw(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor query, Tensor key, Tensor "
      "value, Tensor(a!) initial_state, Tensor state_indices, bool use_qk_l2norm_in_kernel=True) -> Tensor");
  m.def(
      "gdn_conv1d_update(Tensor input, Tensor(a!) conv_state, Tensor weight, Tensor? bias, Tensor "
      "state_indices, bool silu_activation=True) -> Tensor");
  m.def(
      "gdn_conv1d_update.raw(Tensor input, Tensor(a!) conv_state, Tensor weight, Tensor? bias, Tensor "
      "state_indices, bool silu_activation=True) -> Tensor");
  m.def(
      "gdn_packed_decode(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, Tensor(a!) "
      "conv_state, Tensor conv_weight, Tensor? conv_bias, Tensor(b!) recurrent_state, Tensor state_indices, "
      "Tensor(c!) output, bool use_qk_l2norm_in_kernel=True) -> ()");
  m.def(
      "gdn_packed_decode.raw(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, Tensor(a!) "
      "conv_state, Tensor conv_weight, Tensor? conv_bias, Tensor(b!) recurrent_state, Tensor state_indices, "
      "Tensor(c!) output, bool use_qk_l2norm_in_kernel=True) -> ()");
  m.def(
      "gdn_packed_decode_norm(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, Tensor(a!) "
      "conv_state, Tensor conv_weight, Tensor? conv_bias, Tensor(b!) recurrent_state, Tensor state_indices, "
      "Tensor norm_gate, Tensor norm_weight, float norm_epsilon, bool sigmoid_gate, Tensor(c!) output, bool "
      "use_qk_l2norm_in_kernel=True) -> ()");
  m.def(
      "gdn_packed_decode_norm.raw(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor(a!) conv_state, Tensor conv_weight, Tensor? conv_bias, Tensor(b!) recurrent_state, Tensor "
      "state_indices, Tensor norm_gate, Tensor norm_weight, float norm_epsilon, bool sigmoid_gate, "
      "Tensor(c!) output, bool use_qk_l2norm_in_kernel=True) -> ()");
  m.def(
      "gdn_rmsnorm_gated(Tensor input, Tensor weight, Tensor gate, float epsilon, bool sigmoid_gate=False) "
      "-> Tensor");
  m.def(
      "gdn_rmsnorm_gated.raw(Tensor input, Tensor weight, Tensor gate, float epsilon, bool "
      "sigmoid_gate=False) -> Tensor");
  m.def(
      "gdn_prefill(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, Tensor(a!) initial_state, "
      "Tensor cu_seqlens, bool use_qk_l2norm_in_kernel=True) -> Tensor");
  m.def(
      "gdn_prefill.raw(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, Tensor(a!) "
      "initial_state, Tensor cu_seqlens, bool use_qk_l2norm_in_kernel=True) -> Tensor");
  m.def(
      "gdn_conv1d_prefill(Tensor input, Tensor(a!) conv_state, Tensor weight, Tensor? bias, Tensor "
      "query_start_loc, Tensor cache_indices, Tensor has_initial_state, bool silu_activation=True) -> "
      "Tensor");
  m.def(
      "gdn_conv1d_prefill.raw(Tensor input, Tensor(a!) conv_state, Tensor weight, Tensor? bias, Tensor "
      "query_start_loc, Tensor cache_indices, Tensor has_initial_state, bool silu_activation=True) -> "
      "Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_gdn, CPU, m) {
  m.impl("gdn_decode", gdn_decode_cpu);
  m.impl("gdn_decode.raw", gdn_decode_raw);
  m.impl("gdn_conv1d_update", gdn_conv1d_update_cpu);
  m.impl("gdn_conv1d_update.raw", gdn_conv1d_update_raw);
  m.impl("gdn_packed_decode", gdn_packed_decode_cpu);
  m.impl("gdn_packed_decode.raw", gdn_packed_decode_raw);
  m.impl("gdn_packed_decode_norm", flag_gems::arm::gdn_packed_decode_norm_cpu);
  m.impl("gdn_packed_decode_norm.raw", gdn_packed_decode_norm_raw);
  m.impl("gdn_rmsnorm_gated", gdn_rmsnorm_gated_cpu);
  m.impl("gdn_rmsnorm_gated.raw", gdn_rmsnorm_gated_raw);
  m.impl("gdn_prefill", gdn_prefill_cpu);
  m.impl("gdn_prefill.raw", gdn_prefill_raw);
  m.impl("gdn_conv1d_prefill", gdn_conv1d_prefill_cpu);
  m.impl("gdn_conv1d_prefill.raw", gdn_conv1d_prefill_raw);
}
