// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/moments_utils.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <omp.h>

using B = c10::BFloat16;
using BV = at::vec::Vectorized<B>;
using FV = at::vec::Vectorized<float>;

// All callers are workers in the same existing OpenMP team.
extern "C" void flag_gems_arm_add_layer_norm_range(uint64_t address,int64_t begin,int64_t end) {
    const auto* d = reinterpret_cast<const int64_t*>(address);
    auto* y = reinterpret_cast<B*>(d[0]);
    const auto* r = reinterpret_cast<const B*>(d[1]);
    const auto* w = reinterpret_cast<const B*>(d[2]);
    const auto* b = reinterpret_cast<const B*>(d[3]);
    const int64_t rows = d[4], n = d[5];
    const uint32_t bits = uint32_t(d[6]);
    float eps; std::memcpy(&eps, &bits, sizeof(eps));
    alignas(64) B row[4096];
    for (int64_t i=begin; i<end; ++i) {
        int64_t j=0;
        for (; j+BV::size()<=n; j+=BV::size())
            (BV::loadu(y+i*n+j)+BV::loadu(r+i*n+j)).store(row+j);
        for (; j<n; ++j) row[j] = B(float(y[i*n+j])+float(r[i*n+j]));
        auto [mean, variance] = at::native::RowwiseMoments(row, n);
        const float scale = 1.0f/std::sqrt(variance+eps);
        const float shift = -scale*mean;
        j=0;
        for (; j+BV::size()<=n; j+=BV::size()) {
            auto [a0,a1] = at::vec::convert_to_float<B>(BV::loadu(row+j));
            auto [w0,w1] = at::vec::convert_to_float<B>(BV::loadu(w+j));
            auto [b0,b1] = at::vec::convert_to_float<B>(BV::loadu(b+j));
            auto y0 = (a0*FV(scale)+FV(shift))*w0+b0;
            auto y1 = (a1*FV(scale)+FV(shift))*w1+b1;
            at::vec::convert_from_float<B>(y0,y1).store(y+i*n+j);
        }
        for (; j<n; ++j)
            y[i*n+j] = B((float(row[j])*scale+shift)*float(w[j])+float(b[j]));
    }
}

extern "C" void flag_gems_arm_add_layer_norm_rows(uint64_t address) {
    const auto* d=reinterpret_cast<const int64_t*>(address);
    const int64_t rows=d[4],workers=std::min<int64_t>(d[7],omp_get_num_threads());
    const int64_t id=omp_get_thread_num();
    if(id<workers)flag_gems_arm_add_layer_norm_range(address,rows*id/workers,rows*(id+1)/workers);
}
