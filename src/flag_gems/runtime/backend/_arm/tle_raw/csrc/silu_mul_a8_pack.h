// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// CPU FP32 [M,2*K] gate/up -> uint8 qai8dxp panels. K%32=0, MR=1/4.
// Finite input, non-overlapping buffers, inner stride1 are caller invariants.
// A scratch row is reused per calling thread; this function creates no threads.
extern "C" int32_t flag_gems_arm_silu_mul_a8_pack(
    const float* gate_up, uint8_t* packed, int32_t m, int32_t stride, int32_t k, int32_t mr);
