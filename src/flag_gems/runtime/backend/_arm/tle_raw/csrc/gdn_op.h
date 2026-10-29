// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

namespace flag_gems::arm {

void gdn_packed_decode_norm_cpu(const at::Tensor& mixed_qkv,
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
                                bool use_qk_l2norm_in_kernel);

}  // namespace flag_gems::arm
