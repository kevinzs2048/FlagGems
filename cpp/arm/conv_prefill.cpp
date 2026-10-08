// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <omp.h>
#include <torch/library.h>
#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
namespace {
using ConvKernel = void (*)(const void*,
                            void*,
                            const void*,
                            const void*,
                            void*,
                            const void*,
                            const void*,
                            const void*,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t);
std::mutex kernel_mutex;
std::unordered_map<int64_t, ConvKernel> kernels;
int64_t key(int64_t dim, bool bias, bool silu) {
  return dim * 4 + int(bias) * 2 + int(silu);
}
int32_t i32(int64_t value) {
  TORCH_CHECK(value >= 0 && value <= INT32_MAX, "Conv metadata exceeds the CPU int32 ABI");
  return value;
}
at::Tensor conv(const at::Tensor& input,
                at::Tensor& state,
                const at::Tensor& weight,
                const c10::optional<at::Tensor>& bias,
                const at::Tensor& starts,
                const at::Tensor& slots,
                const at::Tensor& initial,
                bool silu) {
  TORCH_CHECK(input.device().is_cpu() && state.device().is_cpu() && weight.device().is_cpu() &&
                  starts.device().is_cpu() && slots.device().is_cpu() && initial.device().is_cpu(),
              "Conv requires CPU tensors");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && state.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "Conv requires BF16 data");
  TORCH_CHECK(input.dim() == 2 && state.dim() == 3 && weight.dim() == 2 && weight.size(1) == 4,
              "Triton conv requires input [dim,tokens], state [slots,dim,history] and weight [dim,4]");
  auto dim = input.size(0), tokens = input.size(1), sequences = starts.numel() - 1;
  TORCH_CHECK(
      dim > 0 && sequences >= 0 && state.size(1) == dim && state.size(2) >= 3 && weight.size(0) == dim,
      "Conv shape mismatch");
  TORCH_CHECK(starts.scalar_type() == at::kInt && slots.scalar_type() == at::kInt && starts.is_contiguous() &&
                  slots.is_contiguous() && slots.numel() == sequences && initial.numel() == sequences,
              "Invalid conv metadata");
  TORCH_CHECK(!bias.has_value() || (bias->device().is_cpu() && bias->scalar_type() == at::kBFloat16 &&
                                    bias->is_contiguous() && bias->numel() == dim),
              "Invalid conv bias");
  auto* begin = starts.const_data_ptr<int32_t>();
  auto* index = slots.const_data_ptr<int32_t>();
  for (int64_t s = 0; s < sequences; ++s)
    TORCH_CHECK(index[s] >= 0 && index[s] < state.size(0) && begin[s] >= 0 && begin[s] <= begin[s + 1] &&
                    begin[s + 1] <= tokens,
                "Conv metadata out of range");
  ConvKernel fn;
  {
    std::lock_guard<std::mutex> guard(kernel_mutex);
    auto it = kernels.find(key(dim, bias.has_value(), silu));
    TORCH_CHECK(it != kernels.end(), "Prepare the Triton conv kernel before inference");
    fn = it->second;
  }
  auto flags = initial.to(at::kBool).contiguous();
  auto output = at::empty_like(input);
  auto blocks = (dim + 63) / 64;
  auto grid = i32(sequences * blocks);
  int32_t is0 = i32(input.stride(0)), is1 = i32(input.stride(1)), ss0 = i32(state.stride(0)),
          ss1 = i32(state.stride(1)), ss2 = i32(state.stride(2));
  int32_t ws0 = i32(weight.stride(0)), ws1 = i32(weight.stride(1)), os0 = i32(output.stride(0)),
          os1 = i32(output.stride(1));
  auto run = [&](int64_t pid) {
    fn(input.const_data_ptr(),
       state.data_ptr(),
       weight.const_data_ptr(),
       bias ? bias->const_data_ptr() : weight.const_data_ptr(),
       output.data_ptr(),
       starts.const_data_ptr(),
       slots.const_data_ptr(),
       flags.const_data_ptr(),
       is0,
       is1,
       ss0,
       ss1,
       ss2,
       ws0,
       ws1,
       os0,
       os1,
       pid,
       0,
       0,
       grid,
       1,
       1);
  };
  // Match the historical CPU Triton launcher's balanced static distribution.
  // ATen's ceil-sized ranges can leave workers idle (128 programs / 18
  // threads becomes 16 active ranges), affecting heterogeneous Arm CPUs.
  // The kernel only touches prepared pointers; no ATen TLS is needed inside.
  const auto workers = std::min<int64_t>(grid, at::get_num_threads());
  if (workers <= 1 || at::in_parallel_region() || omp_in_parallel()) {
    for (int64_t pid = 0; pid < grid; ++pid) run(pid);
  } else {
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int64_t pid = 0; pid < grid; ++pid) run(pid);
  }
  return output;
}
}  // namespace
extern "C" int32_t flag_gems_arm_register_conv_kernel(int32_t dim,
                                                      bool bias,
                                                      bool silu,
                                                      uint64_t address) noexcept {
  if (dim <= 0 || !address) return -1;
  try {
    std::lock_guard<std::mutex> guard(kernel_mutex);
    kernels[key(dim, bias, silu)] = reinterpret_cast<ConvKernel>(address);
    return 0;
  } catch (...) {
    return -1;
  }
}
TORCH_LIBRARY_FRAGMENT(flag_gems_arm_gdn, m) {
  m.def(
      "gdn_conv1d_prefill.triton(Tensor input, Tensor(a!) conv_state, Tensor weight, Tensor? bias, Tensor "
      "query_start_loc, Tensor cache_indices, Tensor has_initial_state, bool silu_activation=True) -> "
      "Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_gdn, CPU, m) {
  m.impl("gdn_conv1d_prefill.triton", conv);
}
