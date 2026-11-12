// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Packed M4 N8 I8MM with an explicit per-channel INT32 accumulation route.
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <atomic>
#include "w8a8_i8mm_tile.h"

template<bool PER_CHANNEL>
static void compute_tile(const int8_t* a, const int8_t* b, const float* xs,
                             const float* ws, float* y, int64_t row, int64_t col,
                             int64_t m, int64_t n, int64_t groups, bool grouped) {
    float32x4_t accum[8];
    for (int i = 0; i < 8; ++i) accum[i] = vdupq_n_f32(0);
    int32x4_t dot[8];
    for (int i = 0; i < 8; ++i) dot[i] = vdupq_n_s32(0);
    for (int64_t g = 0; g < groups; ++g) {
        if constexpr (!PER_CHANNEL) for (int i=0;i<8;++i) dot[i]=vdupq_n_s32(0);
        for (int p = 0; p < 4; ++p) {
            const auto a0 = vld1q_s8(a + (g * 4 + p) * 32);
            const auto a1 = vld1q_s8(a + (g * 4 + p) * 32 + 16);
            for (int c = 0; c < 4; ++c) {
                const auto weight = vld1q_s8(b + (g * 4 + p) * 64 + c * 16);
                dot[c] = vmmlaq_s32(dot[c], a0, weight);
                dot[4 + c] = vmmlaq_s32(dot[4 + c], a1, weight);
            }
        }
        if constexpr (!PER_CHANNEL) for (int r = 0; r < 2; ++r) {
            const int64_t row0 = row + r * 2;
            const float x0 = grouped && row0 < m ? xs[row0 * groups + g] : 1;
            const float x1 = grouped && row0 + 1 < m ? xs[(row0 + 1) * groups + g] : 1;
            const auto xscale = vcombine_f32(vdup_n_f32(x0), vdup_n_f32(x1));
            for (int c = 0; c < 4; ++c) {
                const auto pair = vld1_f32(ws + g * 8 + c * 2);
                const auto scale = vmulq_f32(vcombine_f32(pair, pair), xscale);
                accum[r * 4 + c] = vaddq_f32(accum[r * 4 + c], vmulq_f32(vcvtq_f32_s32(dot[r * 4 + c]), scale));
            }
        }
    }
    if constexpr (PER_CHANNEL) for (int i=0;i<8;++i) accum[i]=vcvtq_f32_s32(dot[i]);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 4; ++c) {
            float values[4];
            vst1q_f32(values, accum[r * 4 + c]);
            for (int rr = 0; rr < 2; ++rr) {
                for (int cc = 0; cc < 2; ++cc) {
                    const auto outrow = row + r * 2 + rr, outcol = col + c * 2 + cc;
                    if (outrow < m && outcol < n) {
                        if constexpr (PER_CHANNEL)
                            y[outrow*n+outcol]=(values[rr*2+cc]*xs[outrow])*ws[c*2+cc];
                        else
                            y[outrow*n+outcol]=values[rr*2+cc]*(grouped?1:xs[outrow]);
                    }
                }
            }
        }
    }
}

// Caller-owned packed tensors; no global cache of source tensor addresses.
extern "C" void flag_gems_arm_w8g32_i8mm_pack(uint64_t address) {
    const auto* d = reinterpret_cast<const int64_t*>(address);
    const auto* w = reinterpret_cast<const int8_t*>(d[0]);
    const auto* scales = reinterpret_cast<const float*>(d[1]);
    auto* packed = reinterpret_cast<int8_t*>(d[2]);
    auto* packed_scales = reinterpret_cast<float*>(d[3]);
    const int64_t n = d[4], k = d[5], groups = k / 32, tiles = (n + 7) / 8;
    const int threads = d[6];
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t g = 0; g < groups; ++g) {
            for (int c = 0; c < 8; ++c) {
                const int64_t col = tile * 8 + c;
                packed_scales[(tile * groups + g) * 8 + c] = col < n ? scales[col * groups + g] : 0;
                for (int p = 0; p < 4; ++p) {
                    auto* dst = packed + ((tile * groups + g) * 4 + p) * 64 + c * 8;
                    if (col < n) std::memcpy(dst, w + col * k + g * 32 + p * 8, 8);
                    else std::memset(dst, 0, 8);
                }
            }
        }
    }
}

// Q, activation scales, packed W, packed scales, Y, M,N,K,threads,grouped.
extern "C" void flag_gems_arm_w8g32_i8mm(uint64_t address) {
    const auto* d = reinterpret_cast<const int64_t*>(address);
    const auto* q = reinterpret_cast<const int8_t*>(d[0]);
    const auto* xs = reinterpret_cast<const float*>(d[1]);
    const auto* w = reinterpret_cast<const int8_t*>(d[2]);
    const auto* ws = reinterpret_cast<const float*>(d[3]);
    auto* y = reinterpret_cast<float*>(d[4]);
    const int64_t m = d[5], n = d[6], k = d[7], groups = k / 32;
    const int threads = d[8];
    const bool grouped = d[9] == 1;
    const bool channel = d[9] == 2;
    const int64_t rows = (m + 3) / 4, cols = (n + 7) / 8;
    thread_local std::vector<int8_t> workspace;
    workspace.resize(rows * groups * 128);
    auto* a = workspace.data();
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static)
        for (int64_t tile = 0; tile < rows; ++tile) {
            for (int64_t g = 0; g < groups; ++g) for (int p = 0; p < 4; ++p) for (int r = 0; r < 4; ++r) {
                auto* dst = a + ((tile * groups + g) * 4 + p) * 32 + r * 8;
                const int64_t row = tile * 4 + r;
                if (row < m) std::memcpy(dst, q + row * k + g * 32 + p * 8, 8);
                else std::memset(dst, 0, 8);
            }
        }
#pragma omp for schedule(static)
        for (int64_t job = 0; job < rows * cols; ++job) {
            const int64_t rt = job / cols, ct = job % cols;
            if (channel)
                compute_tile<true>(a+rt*groups*128,w+ct*groups*256,xs,ws+ct*groups*8,y,rt*4,ct*8,m,n,groups,false);
            else
                compute_tile<false>(a+rt*groups*128,w+ct*groups*256,xs,ws+ct*groups*8,y,rt*4,ct*8,m,n,groups,grouped);
        }
    }
}

extern "C" int flag_gems_arm_w8a8_i8mm_abi_version() { return 2; }

// Each eight-row panel stores two contiguous four-row K streams. A two-worker
// subgroup owns separate cache lines while packing, then consumes the full panel.
// Storage size is unchanged; the shared tile helper keeps its eight-row default
// for the independently packed SME hybrid path.
static inline void pack_panel_rows(const uint16_t* x,int8_t* a,float* xs,
        const float* inverse,int64_t m,int64_t k,int64_t panel,int first,int last){
            for (int rr=first;rr<last;++rr) {
                const int64_t row=panel*8+rr;
                if(row>=m){
                    xs[row]=1.0f;
                    for(int64_t col=0;col<k;col+=8)
                        std::memset(a+panel*k*8+(rr/4)*k*4+(col/8)*32+(rr%4)*8,0,8);
                    continue;
                }
                auto vmax0=vdupq_n_f32(0),vmax1=vdupq_n_f32(0);
                if (row<m) for (int64_t col=0;col<k;col+=8) {
                    const auto bits=vld1q_u16(x+row*k+col);
                    auto low=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits),16));
                    auto high=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits),16));
                    if (inverse) {
                        low=vmulq_f32(low,vld1q_f32(inverse+col));
                        high=vmulq_f32(high,vld1q_f32(inverse+col+4));
                    }
                    vmax0=vmaxq_f32(vmax0,vabsq_f32(low));
                    vmax1=vmaxq_f32(vmax1,vabsq_f32(high));
                }
                const float peak=vmaxvq_f32(vmaxq_f32(vmax0,vmax1));
                const float scale=peak>0 ? peak/127.0f : 1.0f;
                xs[row]=scale;
                for (int64_t col=0;col<k;col+=8) {
                    const auto bits=row<m ? vld1q_u16(x+row*k+col) : vdupq_n_u16(0);
                    auto low=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits),16));
                    auto high=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits),16));
                    if (inverse) {
                        low=vmulq_f32(low,vld1q_f32(inverse+col));
                        high=vmulq_f32(high,vld1q_f32(inverse+col+4));
                    }
                    const auto divisor=vdupq_n_f32(scale);
                    low=vmaxq_f32(vdupq_n_f32(-127),vminq_f32(vdupq_n_f32(127),vdivq_f32(low,divisor)));
                    high=vmaxq_f32(vdupq_n_f32(-127),vminq_f32(vdupq_n_f32(127),vdivq_f32(high,divisor)));
                    const auto q=vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(low)),vqmovn_s32(vcvtnq_s32_f32(high))));
                    vst1_s8(a+panel*k*8+(rr/4)*k*4+(col/8)*32+(rr%4)*8,q);
                }
            }
}
struct alignas(64) PanelReady {
    std::atomic<unsigned> packed{0},computed{0};
};
static inline void panel_wait(std::atomic<unsigned>& count,unsigned workers){
    if(workers==1)return;
    count.fetch_add(1,std::memory_order_acq_rel);
    while(count.load(std::memory_order_acquire)!=workers)asm volatile("yield");
}
extern "C" void flag_gems_arm_add_layer_norm_range(uint64_t,int64_t,int64_t);

// BF16 -> optional FP32 smoothing -> token A8 -> owned I8MM layout -> BF16.
// One region owns complete A panels, then disjoint output tiles. The original
// staged/G32 ABI remains available for reference and other callers.
extern "C" void flag_gems_arm_w8a8_i8mm_fused(uint64_t address) {
    const auto* d=reinterpret_cast<const int64_t*>(address);
    const auto* x=reinterpret_cast<const uint16_t*>(d[0]);
    const auto* w=reinterpret_cast<const int8_t*>(d[1]);
    const auto* ws=reinterpret_cast<const float*>(d[2]);
    auto* a=reinterpret_cast<int8_t*>(d[3]);
    auto* xs=reinterpret_cast<float*>(d[4]);
    auto* y=reinterpret_cast<uint16_t*>(d[5]);
    const auto* bias=reinterpret_cast<const uint16_t*>(d[6]);
    const int64_t m=d[7],n=d[8],k=d[9],mt=(m+7)/8,nt=(n+7)/8;
    const int threads=int(d[10]);
    const auto* inverse=reinterpret_cast<const float*>(d[11]);
    const bool n_major=d[12]!=0;
    auto load=[](const uint16_t* p) {
        return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p),16));
    };
    PanelReady ready[16];
#pragma omp parallel num_threads(threads)
    {
        const int team=omp_get_num_threads();
        if(m>=16 && m<=128 && !n_major && mt>0 && mt<=team && team%mt==0 && team/mt<=8){
            const int workers=team/mt,id=omp_get_thread_num();
            const int64_t panel=id/workers;const int lane=id%workers;
            pack_panel_rows(x,a,xs,inverse,m,k,panel,8*lane/workers,8*(lane+1)/workers);
            panel_wait(ready[panel].packed,workers);
            for(int64_t ni=nt*lane/workers;ni<nt*(lane+1)/workers;++ni)
                w8a8_i8mm_tile8<false,4,16>(a+panel*k*8,w+ni*k*8,xs,ws,bias,y,panel*8,ni*8,m,n,k);
        }else{
#pragma omp for schedule(static) nowait
        for (int64_t panel=0;panel<mt;++panel)
            pack_panel_rows(x,a,xs,inverse,m,k,panel,0,8);
        // Static M-major assignment owns the same whole row panels in
        // packing and GEMM; no other worker reads these panels.
        const int team=omp_get_num_threads();
        const bool row_owned=m<=128 && m%8==0 && !n_major && mt%team==0;
        if(!row_owned){
#pragma omp barrier
        }
        auto compute = [&](int64_t job) {
            const int64_t mi=n_major ? job%mt : job/nt;
            const int64_t ni=n_major ? job/mt : job%nt;
            w8a8_i8mm_tile8<false,4,16>(a+mi*k*8,w+ni*k*8,xs,ws,bias,y,mi*8,ni*8,m,n,k);
        };
        // Partial row tiles cost less than full tiles. Balance only these
        // short-input cases; aligned matrices retain the validated static route.
        if(m>=16 && m<=128 && m%8>0 && m%8<=4 && k<=2048){
#pragma omp for schedule(dynamic,16) nowait
            for(int64_t job=0;job<mt*nt;++job)compute(job);
        }else if(m>=16 && m<=128 && m%8>0 && m%8<=4){
#pragma omp for schedule(dynamic,8) nowait
            for(int64_t job=0;job<mt*nt;++job)compute(job);
        }else
#if FLAGGEMS_I8MM_DYNAMIC_CHUNK > 0
        // Small M benefits from static assignment. Large M needs finer
        // balancing across the larger output-tile workload.
        if (m > 128 || (m > 64 && n_major)) {
#pragma omp for schedule(dynamic,FLAGGEMS_I8MM_DYNAMIC_CHUNK) nowait
            for (int64_t job=0;job<mt*nt;++job) compute(job);
        } else
#endif
        {
#pragma omp for schedule(static) nowait
            for (int64_t job=0;job<mt*nt;++job) compute(job);
        }
        }
    }
}

extern "C" void flag_gems_arm_add_layer_norm_rows(uint64_t address);
extern "C" int flag_gems_arm_w8a8_i8mm_norm_abi_version() { return 1; }
extern "C" void flag_gems_arm_w8a8_i8mm_fused_norm(uint64_t address) {
    const auto* d=reinterpret_cast<const int64_t*>(address);
    const auto* x=reinterpret_cast<const uint16_t*>(d[0]);
    const auto* w=reinterpret_cast<const int8_t*>(d[1]);
    const auto* ws=reinterpret_cast<const float*>(d[2]);
    auto* a=reinterpret_cast<int8_t*>(d[3]);
    auto* xs=reinterpret_cast<float*>(d[4]);
    auto* y=reinterpret_cast<uint16_t*>(d[5]);
    const auto* bias=reinterpret_cast<const uint16_t*>(d[6]);
    const int64_t m=d[7],n=d[8],k=d[9],mt=(m+7)/8,nt=(n+7)/8;
    const int threads=int(d[10]);
    const auto* inverse=reinterpret_cast<const float*>(d[11]);
    const bool n_major=d[12]!=0;
    auto load=[](const uint16_t* p) {
        return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p),16));
    };
    PanelReady ready[16];
#pragma omp parallel num_threads(threads)
    {
        const int team=omp_get_num_threads();
        if(m>=16 && m<=128 && !n_major && mt>0 && mt<=team && team%mt==0 && team/mt<=8 && d[17]>=team){
            const int workers=team/mt,id=omp_get_thread_num();
            const int64_t panel=id/workers;const int lane=id%workers;
            pack_panel_rows(x,a,xs,inverse,m,k,panel,8*lane/workers,8*(lane+1)/workers);
            panel_wait(ready[panel].packed,workers);
            for(int64_t ni=nt*lane/workers;ni<nt*(lane+1)/workers;++ni)
                w8a8_i8mm_tile8<false,4,16>(a+panel*k*8,w+ni*k*8,xs,ws,bias,y,panel*8,ni*8,m,n,k);
            panel_wait(ready[panel].computed,workers);
            const int64_t count=std::min<int64_t>(8,m-panel*8);
            int64_t nd[8]={int64_t(y),d[13],d[14],d[15],m,n,d[16],d[17]};
            flag_gems_arm_add_layer_norm_range(uint64_t(nd),panel*8+count*lane/workers,panel*8+count*(lane+1)/workers);
        }else{
#pragma omp for schedule(static) nowait
        for (int64_t panel=0;panel<mt;++panel)
            pack_panel_rows(x,a,xs,inverse,m,k,panel,0,8);
        // Static M-major assignment owns the same whole row panels in
        // packing and GEMM; no other worker reads these panels.
        const int team=omp_get_num_threads();
        const bool row_owned=m<=128 && m%8==0 && !n_major && mt%team==0;
        if(!row_owned){
#pragma omp barrier
        }
        auto compute = [&](int64_t job) {
            const int64_t mi=n_major ? job%mt : job/nt;
            const int64_t ni=n_major ? job/mt : job%nt;
            w8a8_i8mm_tile8<false,4,16>(a+mi*k*8,w+ni*k*8,xs,ws,bias,y,mi*8,ni*8,m,n,k);
        };
        // Partial row tiles cost less than full tiles. Balance only these
        // short-input cases; aligned matrices retain the validated static route.
        if(m>=16 && m<=128 && m%8>0 && m%8<=4 && k<=2048){
#pragma omp for schedule(dynamic,16) nowait
            for(int64_t job=0;job<mt*nt;++job)compute(job);
        }else if(m>=16 && m<=128 && m%8>0 && m%8<=4){
#pragma omp for schedule(dynamic,8) nowait
            for(int64_t job=0;job<mt*nt;++job)compute(job);
        }else
#if FLAGGEMS_I8MM_DYNAMIC_CHUNK > 0
        // Small M benefits from static assignment. Large M needs finer
        // balancing across the larger output-tile workload.
        if (m > 128 || (m > 64 && n_major)) {
#pragma omp for schedule(dynamic,FLAGGEMS_I8MM_DYNAMIC_CHUNK) nowait
            for (int64_t job=0;job<mt*nt;++job) compute(job);
        } else
#endif
        {
#pragma omp for schedule(static) nowait
            for (int64_t job=0;job<mt*nt;++job) compute(job);
        }
        // Normalization may run immediately only when its row partition
        // matches the complete GEMM panels owned by this worker.
        if(!row_owned || d[17]<team){
#pragma omp barrier
        }
        int64_t norm_descriptor[8]={int64_t(y),d[13],d[14],d[15],m,n,d[16],d[17]};
        flag_gems_arm_add_layer_norm_rows(uint64_t(norm_descriptor));
        }
    }
}


// A prepared BF16 activation table preserves the caller's activation rounding.
// Each subgroup owns one row panel through both GEMMs and the final norm.
struct alignas(64) FFNPanelReady {
    std::atomic<unsigned> packed{0},expanded{0},activated{0},projected{0};
};
static void fused_ffn(const int64_t* up,const int64_t* down,const uint16_t* lut) {
    const int64_t m=up[7],h=up[8],k=up[9],mt=(m+7)/8;
    auto* hidden=reinterpret_cast<uint16_t*>(up[5]);
    FFNPanelReady ready[16];
    // Non-divisible row panels retain the validated, fully occupied route.
    const int threads=int(up[10]);
    if(m<16 || m>128 || up[12] || down[12] || mt>threads || threads%mt || down[17]<threads){
        flag_gems_arm_w8a8_i8mm_fused(uint64_t(up));
#pragma omp parallel for num_threads(threads) schedule(static)
        for(int64_t i=0;i<m*h;++i)hidden[i]=lut[hidden[i]];
        flag_gems_arm_w8a8_i8mm_fused_norm(uint64_t(down));
        return;
    }
#pragma omp parallel num_threads(threads)
    {
        const int team=omp_get_num_threads();
        // OpenMP may shrink the team under resource limits. Work-sharing
        // fallback remains correct without assuming the requested team size.
        if(mt<=team && team%mt==0 && team/mt<=8){
            const int workers=team/mt,id=omp_get_thread_num(),lane=id%workers;
            const int64_t panel=id/workers,first=8*lane/workers,last=8*(lane+1)/workers;
            pack_panel_rows(reinterpret_cast<const uint16_t*>(up[0]),reinterpret_cast<int8_t*>(up[3]),
                reinterpret_cast<float*>(up[4]),reinterpret_cast<const float*>(up[11]),m,k,panel,first,last);
            panel_wait(ready[panel].packed,workers);
            for(int64_t ni=((h+7)/8)*lane/workers;ni<((h+7)/8)*(lane+1)/workers;++ni)
                w8a8_i8mm_tile8<true,4,16>(reinterpret_cast<const int8_t*>(up[3])+panel*k*8,
                    reinterpret_cast<const int8_t*>(up[1])+ni*k*8,reinterpret_cast<const float*>(up[4]),
                    reinterpret_cast<const float*>(up[2]),reinterpret_cast<const uint16_t*>(up[6]),hidden,panel*8,ni*8,m,h,k,lut);
            panel_wait(ready[panel].expanded,workers);
            pack_panel_rows(hidden,reinterpret_cast<int8_t*>(down[3]),reinterpret_cast<float*>(down[4]),
                reinterpret_cast<const float*>(down[11]),m,h,panel,first,last);
            panel_wait(ready[panel].activated,workers);
            auto* output=reinterpret_cast<uint16_t*>(down[5]);
            for(int64_t ni=((k+7)/8)*lane/workers;ni<((k+7)/8)*(lane+1)/workers;++ni)
                w8a8_i8mm_tile8<false,4,16>(reinterpret_cast<const int8_t*>(down[3])+panel*h*8,
                    reinterpret_cast<const int8_t*>(down[1])+ni*h*8,reinterpret_cast<const float*>(down[4]),
                    reinterpret_cast<const float*>(down[2]),reinterpret_cast<const uint16_t*>(down[6]),output,panel*8,ni*8,m,k,h);
            panel_wait(ready[panel].projected,workers);
            const int64_t count=std::min<int64_t>(8,m-panel*8);
            int64_t nd[8]={down[5],down[13],down[14],down[15],m,k,down[16],down[17]};
            flag_gems_arm_add_layer_norm_range(uint64_t(nd),panel*8+count*lane/workers,panel*8+count*(lane+1)/workers);
        }else{
#pragma omp single
            {
                // Called from the single worker only; inner regions are valid
                // even when OpenMP serializes nested teams.
                flag_gems_arm_w8a8_i8mm_fused(uint64_t(up));
                for(int64_t i=0;i<m*h;++i)hidden[i]=lut[hidden[i]];
                flag_gems_arm_w8a8_i8mm_fused_norm(uint64_t(down));
            }
        }
    }
}

#ifdef FLAGGEMS_ARM_I8MM_TENSOR_RAW
#include "raw_dispatch.h"
extern "C" int flag_gems_arm_w8a8_i8mm_ffn_abi_version() { return 1; }
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <optional>
#include <cmath>
#include <c10/util/SmallVector.h>
#include <thread>

namespace {
using OptionalTensor = std::optional<at::Tensor>;
void check_tensor(const at::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==dtype && t.is_contiguous(),
                "I8MM Raw requires contiguous CPU tensors with matching dtypes");
}
at::Tensor tensor_i8mm(const at::Tensor& input,const at::Tensor& packed,
        const at::Tensor& scales,const OptionalTensor& inverse,const OptionalTensor& bias,
        int64_t threads,bool n_major,const OptionalTensor& residual,
        const OptionalTensor& norm_weight,const OptionalTensor& norm_bias,
        int64_t eps_bits,int64_t norm_threads) {
    TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()>=1,
                "I8MM Raw requires CPU BF16 input with rank >=1");
    check_tensor(packed,at::kChar);check_tensor(scales,at::kFloat);
    const int64_t k=input.size(-1),n=scales.numel();
    TORCH_CHECK(k>0 && k%32==0 && n>0 && scales.dim()==1 && packed.dim()==4,
                "Invalid I8MM dimensions");
    TORCH_CHECK(packed.size(0)==(n+7)/8 && packed.size(1)==k/8 && packed.size(2)==8 && packed.size(3)==8,
                "I8MM packed tensor shape does not match input/scales");
    static const auto thread_limit=std::max<unsigned>(1,std::thread::hardware_concurrency());
    TORCH_CHECK(threads>0 && threads<=thread_limit,"Invalid I8MM threads");
    if(inverse){check_tensor(*inverse,at::kFloat);TORCH_CHECK(inverse->dim()==1 && inverse->numel()==k,"Invalid smoothing shape");}
    if(bias){check_tensor(*bias,at::kBFloat16);TORCH_CHECK(bias->dim()==1 && bias->numel()==n,"Invalid bias shape");}
    if(at::GradMode::is_enabled())
        TORCH_CHECK(!input.requires_grad() && (!bias || !bias->requires_grad()),"I8MM Raw is inference only");
    c10::SmallVector<int64_t,4> shape(input.sizes().begin(),input.sizes().end());shape.back()=n;
    if(residual){
        TORCH_CHECK(input.dim()>=2 && n<=4096 && norm_weight && norm_bias,"Invalid norm dimensions");
        check_tensor(*residual,at::kBFloat16);check_tensor(*norm_weight,at::kBFloat16);check_tensor(*norm_bias,at::kBFloat16);
        TORCH_CHECK(residual->sizes()==at::IntArrayRef(shape) && norm_weight->dim()==1 && norm_weight->numel()==n && norm_bias->sizes()==norm_weight->sizes(),"Invalid residual/norm shape");
        TORCH_CHECK(norm_threads>0 && norm_threads<=thread_limit,"Invalid norm threads");
        const uint32_t bits=uint32_t(eps_bits);float epsilon;std::memcpy(&epsilon,&bits,sizeof(epsilon));
        TORCH_CHECK(std::isfinite(epsilon) && epsilon>0,"Invalid norm epsilon");
        if(at::GradMode::is_enabled())TORCH_CHECK(!residual->requires_grad() && !norm_weight->requires_grad() && !norm_bias->requires_grad(),"I8MM Raw is inference only");
    }
    auto flat=input.dim()==2 && input.is_contiguous() ? input : input.reshape({-1,k}).contiguous();
    const int64_t rows=flat.size(0),padded=(rows+7)/8*8;
    auto output=at::empty({rows,n},input.options());
    if(rows){
        thread_local at::Tensor activations,activation_scales;
        if(!activations.defined() || activations.numel()<padded*k)activations=at::empty({padded*k},input.options().dtype(at::kChar));
        if(!activation_scales.defined() || activation_scales.numel()<padded)activation_scales=at::empty({padded},input.options().dtype(at::kFloat));
        int64_t d[18]={reinterpret_cast<int64_t>(flat.const_data_ptr()),reinterpret_cast<int64_t>(packed.const_data_ptr()),
            reinterpret_cast<int64_t>(scales.const_data_ptr()),reinterpret_cast<int64_t>(activations.mutable_data_ptr()),
            reinterpret_cast<int64_t>(activation_scales.mutable_data_ptr()),reinterpret_cast<int64_t>(output.mutable_data_ptr()),
            bias?reinterpret_cast<int64_t>(bias->const_data_ptr()):0,rows,n,k,threads,
            inverse?reinterpret_cast<int64_t>(inverse->const_data_ptr()):0,int64_t(n_major),0,0,0,eps_bits,norm_threads};
        if(residual){
            d[13]=reinterpret_cast<int64_t>(residual->const_data_ptr());
            d[14]=reinterpret_cast<int64_t>(norm_weight->const_data_ptr());
            d[15]=reinterpret_cast<int64_t>(norm_bias->const_data_ptr());
            flag_gems_arm_w8a8_i8mm_fused_norm(reinterpret_cast<uint64_t>(d));
        }else flag_gems_arm_w8a8_i8mm_fused(reinterpret_cast<uint64_t>(d));
    }
    return input.dim()==2 ? output : output.reshape(shape);
}
at::Tensor tensor_linear_raw(const at::Tensor& x,const at::Tensor& w,const at::Tensor& scales,
        const OptionalTensor& inverse,const OptionalTensor& bias,int64_t threads,bool n_major) {
    return flag_gems::arm::invoke_raw([&]{return tensor_i8mm(x,w,scales,inverse,bias,threads,n_major,std::nullopt,std::nullopt,std::nullopt,0,1);});
}
at::Tensor tensor_norm_raw(const at::Tensor& x,const at::Tensor& residual,const at::Tensor& w,
        const at::Tensor& scales,const OptionalTensor& inverse,const OptionalTensor& bias,
        const at::Tensor& nw,const at::Tensor& nb,int64_t eps_bits,int64_t threads,int64_t norm_threads,bool n_major) {
    return flag_gems::arm::invoke_raw([&]{return tensor_i8mm(x,w,scales,inverse,bias,threads,n_major,residual,nw,nb,eps_bits,norm_threads);});
}
at::Tensor tensor_ffn_raw(const at::Tensor& input,const at::Tensor& uw,const at::Tensor& us,
        const OptionalTensor& ui,const OptionalTensor& ub,const at::Tensor& dw,const at::Tensor& ds,
        const OptionalTensor& di,const OptionalTensor& db,const at::Tensor& lut,
        const at::Tensor& nw,const at::Tensor& nb,int64_t eps_bits,int64_t threads,
        int64_t norm_threads,bool up_n_major,bool down_n_major) {
    return flag_gems::arm::invoke_raw([&]{
        TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()>=2,
                    "FFN Raw requires CPU BF16 input with rank >=2");
        const int64_t k=input.size(-1),h=us.numel();
        TORCH_CHECK(k>0 && k<=4096 && k%32==0 && h>0 && h%32==0,"Invalid FFN dimensions");
        static const auto limit=std::max<unsigned>(1,std::thread::hardware_concurrency());
        TORCH_CHECK(threads>0 && threads<=limit && norm_threads>0 && norm_threads<=limit,"Invalid FFN threads");
        auto check_plan=[&](const at::Tensor& w,const at::Tensor& scale,const OptionalTensor& inverse,
                const OptionalTensor& bias,int64_t n,int64_t inner){
            check_tensor(w,at::kChar);check_tensor(scale,at::kFloat);
            TORCH_CHECK(scale.dim()==1 && scale.numel()==n && w.dim()==4 && w.size(0)==(n+7)/8
                && w.size(1)==inner/8 && w.size(2)==8 && w.size(3)==8,"Invalid FFN packed shape");
            if(inverse){check_tensor(*inverse,at::kFloat);TORCH_CHECK(inverse->dim()==1 && inverse->numel()==inner,"Invalid FFN smoothing");}
            if(bias){check_tensor(*bias,at::kBFloat16);TORCH_CHECK(bias->dim()==1 && bias->numel()==n,"Invalid FFN bias");}
        };
        check_plan(uw,us,ui,ub,h,k);check_plan(dw,ds,di,db,k,h);
        check_tensor(lut,at::kBFloat16);check_tensor(nw,at::kBFloat16);check_tensor(nb,at::kBFloat16);
        TORCH_CHECK(lut.dim()==1 && lut.numel()==65536 && nw.dim()==1 && nw.numel()==k && nb.sizes()==nw.sizes(),"Invalid FFN activation/norm shape");
        const uint32_t bits=uint32_t(eps_bits);float eps;std::memcpy(&eps,&bits,4);
        TORCH_CHECK(std::isfinite(eps) && eps>0,"Invalid FFN epsilon");
        if(at::GradMode::is_enabled()){
            for(const auto& t:{input,nw,nb,lut})TORCH_CHECK(!t.requires_grad(),"FFN Raw is inference only");
            TORCH_CHECK((!ub || !ub->requires_grad()) && (!db || !db->requires_grad()),"FFN Raw is inference only");
        }
        auto x=input.dim()==2 && input.is_contiguous() ? input : input.reshape({-1,k}).contiguous();
        const int64_t m=x.size(0),mp=(m+7)/8*8;
        auto output=at::empty(x.sizes(),input.options());
        if(m){
            thread_local at::Tensor hidden,au,su,ad,sd;
            auto reserve=[&](at::Tensor& t,int64_t count,at::ScalarType dtype){
                if(!t.defined() || t.numel()<count)t=at::empty({count},input.options().dtype(dtype));
            };
            reserve(hidden,m*h,at::kBFloat16);reserve(au,mp*k,at::kChar);reserve(su,mp,at::kFloat);
            reserve(ad,mp*h,at::kChar);reserve(sd,mp,at::kFloat);
            auto ptr=[](const at::Tensor& t){return reinterpret_cast<int64_t>(t.const_data_ptr());};
            int64_t up[13]={ptr(x),ptr(uw),ptr(us),ptr(au),ptr(su),ptr(hidden),ub?ptr(*ub):0,m,h,k,threads,ui?ptr(*ui):0,int64_t(up_n_major)};
            int64_t down[18]={ptr(hidden),ptr(dw),ptr(ds),ptr(ad),ptr(sd),ptr(output),db?ptr(*db):0,m,k,h,threads,di?ptr(*di):0,int64_t(down_n_major),ptr(x),ptr(nw),ptr(nb),eps_bits,norm_threads};
            fused_ffn(up,down,reinterpret_cast<const uint16_t*>(lut.const_data_ptr()));
        }
        return input.dim()==2 ? output : output.reshape(input.sizes());
    });
}

}
TORCH_LIBRARY_FRAGMENT(flag_gems_arm_w8a8_i8mm,m) {
    m.def("ffn.raw(Tensor x, Tensor up_packed, Tensor up_scales, Tensor? up_inverse, Tensor? up_bias, Tensor down_packed, Tensor down_scales, Tensor? down_inverse, Tensor? down_bias, Tensor activation_lut, Tensor norm_weight, Tensor norm_bias, int eps_bits, int threads, int norm_threads, bool up_n_major, bool down_n_major) -> Tensor");
    m.def("linear.raw(Tensor x, Tensor packed, Tensor scales, Tensor? inverse, Tensor? bias, int threads, bool n_major) -> Tensor");
    m.def("add_norm.raw(Tensor x, Tensor residual, Tensor packed, Tensor scales, Tensor? inverse, Tensor? bias, Tensor norm_weight, Tensor norm_bias, int eps_bits, int threads, int norm_threads, bool n_major) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_w8a8_i8mm,CPU,m) {
    m.impl("ffn.raw",tensor_ffn_raw);
    m.impl("linear.raw",tensor_linear_raw);m.impl("add_norm.raw",tensor_norm_raw);
}
#endif
