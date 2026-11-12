// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Symmetric per-channel W8, dynamic per-token A8; INT32 SME2 accumulation.
#include <arm_sme.h>
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include "w8a8_i8mm_tile.h"
#if defined(FLAGGEMS_W8_FFN_ONLY)
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/Gelu.h>
#include <array>
#endif
extern "C" void flag_gems_arm_add_layer_norm_rows(uint64_t address);

static float bf16_float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

static uint16_t float_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 0x40u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static float32x4_t load_bf16x4(const uint16_t* value) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(value),16));
}

#if defined(FLAGGEMS_W8_SVE_PACK)
// Experimental M5 packer: preserve FP32 division and nearest-even rounding.
// The existing Neon path remains available for an isolated paired comparison.
template<bool HYBRID>
__arm_locally_streaming
static void pack_block_sve(const uint16_t* x, const float* inverse,
                          const float* xs, int8_t* a, int8_t* ai,
                          int64_t m, int64_t k, int64_t panel,
                          int64_t start, int64_t end) {
    if (svcntsw() != 16) __builtin_trap();
    for (int64_t row=panel*16; row<(panel+1)*16; ++row) {
        for (int64_t col=start; col<end; col+=16) {
            const auto pg=svwhilelt_b32(col,end);
            auto value=svdup_n_f32(0);
            if (row<m)
                value=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,x+row*k+col),16));
            if (inverse) value=svmul_f32_x(pg,value,svld1_f32(pg,inverse+col));
            value=svdiv_n_f32_x(pg,value,xs[row]);
            value=svmax_n_f32_x(pg,svmin_n_f32_x(pg,value,127),-127);
            const auto quant=svcvt_s32_f32_x(pg,svrintn_f32_x(pg,value));
            int8_t bytes[16];
            svst1b_s32(pg,bytes,quant);
            for (int64_t offset=0;offset<std::min<int64_t>(16,end-col);offset+=4) {
                const int64_t c=col+offset;
                std::memcpy(a+(row/16)*k*16+(c/4)*64+(row%16)*4,bytes+offset,4);
                if constexpr (HYBRID)
                    std::memcpy(ai+(row/8)*k*8+(c/8)*64+(row%8/2)*16+(row%2)*8+c%8,bytes+offset,4);
            }
        }
    }
}
#endif

template<bool FUSED,bool DYNAMIC=false,bool TALL=false,bool SPLIT_QUEUE=false>
__arm_new("za") __arm_locally_streaming
static int64_t compute(const int8_t* a, const int8_t* w, int32_t* out,
                    int64_t k, int64_t np, int64_t mt, int64_t begin, int64_t end,
                    const float* xs,const float* ws,const uint16_t* bias,
                    uint16_t* y,int64_t m,int64_t n,int64_t* next=nullptr) {
#if defined(FLAGGEMS_W8_M32_TILE)
    if constexpr (!TALL)
        if (m==32) return compute<FUSED,DYNAMIC,true>(a,w,out,k,np,mt,begin,end,xs,ws,bias,y,m,n,next);
#endif
    if (svcntsw() != 16) __builtin_trap();
    const auto pb = svptrue_b8();
    const auto pi = svptrue_b32();
    int64_t processed=0;
    auto claim_sme=[&] {
        if constexpr(SPLIT_QUEUE) {
            auto* queue=reinterpret_cast<uint64_t*>(next);
            uint64_t state=__atomic_load_n(queue,__ATOMIC_RELAXED);
            for(;;) {
                const uint32_t front=uint32_t(state),back=uint32_t(state>>32);
                if(back-front<2)return end;
                const uint64_t desired=(uint64_t(back)<<32)|uint64_t(front+2);
                if(__atomic_compare_exchange_n(queue,&state,desired,true,__ATOMIC_RELAXED,__ATOMIC_RELAXED))
                    return int64_t(front/2);
            }
        }
        return __atomic_fetch_add(next,1,__ATOMIC_RELAXED);
    };
    for (int64_t job=DYNAMIC?claim_sme():begin;
         job<end;job=DYNAMIC?claim_sme():job+1) {
        constexpr int TM=TALL?32:16,TN=TALL?32:64;
        const int64_t tiles_m=TALL?1:mt;
        const int64_t mi = job % tiles_m, ni = job / tiles_m;
        const auto* ap = a + mi*k*TM;
        const auto* wp = TALL ? w+(ni/2)*k*64+(ni%2)*128 : w+ni*k*64;
        svzero_za();
#if defined(FLAGGEMS_W8_SME_UNROLL4)
#pragma clang loop unroll_count(4)
#endif
        for (int64_t group = 0; group < k/4; ++group) {
            const auto av = svld1_s8(pb, ap + group*64);
            const auto* b = wp + group*256;
            const auto b0=svld1_s8(pb,b), b1=svld1_s8(pb,b+64);
            svmopa_za32_s8_m(0, pb, pb, av, b0);
            svmopa_za32_s8_m(1, pb, pb, av, b1);
            if constexpr (TALL) {
                const auto av1=svld1_s8(pb,ap+k*16+group*64);
                svmopa_za32_s8_m(2,pb,pb,av1,b0);
                svmopa_za32_s8_m(3,pb,pb,av1,b1);
            } else {
                svmopa_za32_s8_m(2,pb,pb,av,svld1_s8(pb,b+128));
                svmopa_za32_s8_m(3,pb,pb,av,svld1_s8(pb,b+192));
            }
        }
#if defined(FLAGGEMS_W8_SME_HOIST_EPILOGUE)
#define STORE_CACHE(C) \
        auto cached_ws=svdup_n_f32(0),cached_bias=svdup_n_f32(0); \
        if constexpr(FUSED) { \
            const int64_t cc=ni*TN+(C); \
            const auto cp=svwhilelt_b32(cc,n); \
            cached_ws=svld1_f32(cp,ws+cc); \
            if(bias)cached_bias=svreinterpret_f32_u32(svlsl_n_u32_x(cp,svld1uh_u32(cp,bias+cc),16)); \
        }
#define STORE_WS(PG,COL) cached_ws
#define STORE_BIAS(PG,COL) cached_bias
#else
#define STORE_CACHE(C)
#define STORE_WS(PG,COL) svld1_f32(PG,ws+COL)
#define STORE_BIAS(PG,COL) svreinterpret_f32_u32(svlsl_n_u32_x(PG,svld1uh_u32(PG,bias+COL),16))
#endif
#define STORE_ROW(V,R,C) \
        if constexpr (FUSED) { \
            const int64_t row=mi*TM+(R),col=ni*TN+(C); \
            if (row<m) { \
                const auto pg=svwhilelt_b32(col,n); \
                auto value=svmul_n_f32_x(pg,svcvt_f32_s32_x(pg,V),xs[row]); \
                value=svmul_f32_x(pg,value,STORE_WS(pg,col)); \
                if (bias) value=svadd_f32_x(pg,value,STORE_BIAS(pg,col)); \
                const auto bits=svreinterpret_u32_f32(value); \
                const auto rounding=svadd_n_u32_x(pg,svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1),0x7fff); \
                auto rounded=svlsr_n_u32_x(pg,svadd_u32_x(pg,bits,rounding),16); \
                const auto nan=svcmpgt_n_u32(pg,svand_n_u32_x(pg,bits,0x7fffffff),0x7f800000); \
                rounded=svsel_u32(nan,svorr_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),0x40),rounded); \
                svst1h_u32(pg,y+row*n+col,rounded); \
            } \
        } else { \
            svst1_s32(pi,out+(mi*TM+(R))*np+ni*TN+(C),V); \
        }
#define STORE(T,R,C) \
    { STORE_CACHE(C) \
        for (int r=0; r<16; r+=4) { \
            const auto four=svread_hor_za32_s32_vg4(T,r); \
            STORE_ROW(svget4_s32(four,0),(R)+r,C) \
            STORE_ROW(svget4_s32(four,1),(R)+r+1,C) \
            STORE_ROW(svget4_s32(four,2),(R)+r+2,C) \
            STORE_ROW(svget4_s32(four,3),(R)+r+3,C) \
        } \
    }
#if defined(FLAGGEMS_W8_SME_ROW_STORE)
// Write all column slices for four rows before moving to the next rows.
// This preserves every scale/rounding operation, reducing the distance between
// writes to adjacent portions of the same output row for short inputs.
#define STORE_QUAD(T,R,C) \
    { STORE_CACHE(C) \
        const auto four=svread_hor_za32_s32_vg4(T,r); \
        STORE_ROW(svget4_s32(four,0),(R)+r,C) \
        STORE_ROW(svget4_s32(four,1),(R)+r+1,C) \
        STORE_ROW(svget4_s32(four,2),(R)+r+2,C) \
        STORE_ROW(svget4_s32(four,3),(R)+r+3,C) \
    }
        if(m<=64){
            for(int r=0;r<16;r+=4){
                STORE_QUAD(0,0,0)
                STORE_QUAD(1,0,16)
                if constexpr(TALL){
                    STORE_QUAD(2,16,0)
                    STORE_QUAD(3,16,16)
                }else{
                    STORE_QUAD(2,0,32)
                    STORE_QUAD(3,0,48)
                }
            }
        }else
#undef STORE_QUAD
#endif
        {
        STORE(0,0,0)
        STORE(1,0,16)
        if constexpr (TALL) {
            STORE(2,16,0)
            STORE(3,16,16)
        } else {
            STORE(2,0,32)
            STORE(3,0,48)
        }
        }
#undef STORE
#undef STORE_ROW
#undef STORE_CACHE
#undef STORE_WS
#undef STORE_BIAS
        ++processed;
    }
    return processed;
}

template<bool FUSED=false,bool HYBRID=false,bool NORM=false,bool IN_TEAM=false,bool PRE_SCALED=false>
static void run(uint64_t address, bool scaled, bool profile=false, int64_t* shared_next=nullptr) {
    const auto* d=reinterpret_cast<const int64_t*>(address);
    const auto* x=reinterpret_cast<const uint16_t*>(d[0]);
    const auto* w=reinterpret_cast<const int8_t*>(d[1]);
    const auto* ws=reinterpret_cast<const float*>(d[2]);
    auto* a=reinterpret_cast<int8_t*>(d[3]);
    auto* xs=reinterpret_cast<float*>(d[4]);
    auto* scratch=reinterpret_cast<int32_t*>(d[5]);
    auto* y=reinterpret_cast<uint16_t*>(d[6]);
    const auto* bias=reinterpret_cast<const uint16_t*>(d[7]);
    constexpr int TM=16,TN=64;
    const int64_t m=d[8], n=d[9], k=d[10], mt=(m+TM-1)/TM, nt=(n+TN-1)/TN, np=(n+63)/64*64;
    const int threads=int(d[11]);
    const auto* inverse=scaled ? reinterpret_cast<const float*>(d[12]) : nullptr;
    const auto* wi=HYBRID?reinterpret_cast<const int8_t*>(d[14]):nullptr;
    auto* ai=HYBRID?reinterpret_cast<int8_t*>(d[15]):nullptr;
    const int sme_threads=HYBRID?int(d[16]):threads;
    int64_t next=0;
    auto* next_ptr=shared_next ? shared_next : &next;
    int64_t route_counts[2]={0,0};
    auto* times=profile?reinterpret_cast<double*>(d[13]):nullptr;
    const double started=profile?omp_get_wtime():0;
    auto worker=[&] {
        // One shared double-ended queue: SME claims aligned M16xN64 tiles
        // from the front, Neon claims M8xN64 halves from the back. CAS updates
        // both boundaries atomically, so no cell has two owners. Both paths
        // keep full output cache lines and the existing weight/A8 layouts.
        const bool split_queue=HYBRID && m==32 && sme_threads>0
            && omp_get_num_threads()>sme_threads && mt*nt<INT32_MAX/2;
#pragma omp master
        {
            if(split_queue) __atomic_store_n(reinterpret_cast<uint64_t*>(next_ptr),
                uint64_t(mt*nt*2)<<32,__ATOMIC_RELAXED);
        }
        // Packing has an implicit barrier before any queue consumer.
        // First establish one scale per row. Then assign whole cache-line
        // blocks of both A8 layouts to workers, avoiding shared-line writes.
if constexpr(!PRE_SCALED) {
#pragma omp for schedule(static)
        for (int64_t row=0;row<mt*TM;++row) {
            auto vmax=vdupq_n_f32(0);
            if (row<m) for (int64_t col=0;col<k;col+=4) {
                auto value=load_bf16x4(x+row*k+col);
                if (inverse) value=vmulq_f32(value,vld1q_f32(inverse+col));
                vmax=vmaxq_f32(vmax,vabsq_f32(value));
            }
            xs[row]=std::max(vmaxvq_f32(vmax)/127.0f,1e-8f);
        }
        }
        constexpr int64_t KC=256;
        const int64_t kt=(k+KC-1)/KC;
#pragma omp for schedule(static)
        for (int64_t job=0;job<mt*kt;++job) {
            const int64_t panel=job/kt,start=(job%kt)*KC,end=std::min(k,start+KC);
#if defined(FLAGGEMS_W8_SVE_PACK)
            pack_block_sve<HYBRID>(x,inverse,xs,a,ai,m,k,panel,start,end);
#else
            for (int64_t row=panel*TM;row<(panel+1)*TM;++row) {
                const auto vs=vdupq_n_f32(xs[row]);
                int64_t col=start;
#if defined(FLAGGEMS_W8_PACK8)
                if constexpr(HYBRID) {
                    for(;col+8<=end;col+=8) {
                        const auto bits=row<m?vld1q_u16(x+row*k+col):vdupq_n_u16(0);
                        auto lo=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits),16));
                        auto hi=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits),16));
                        if(inverse) {
                            lo=vmulq_f32(lo,vld1q_f32(inverse+col));
                            hi=vmulq_f32(hi,vld1q_f32(inverse+col+4));
                        }
                        lo=vmaxq_f32(vdupq_n_f32(-127),vminq_f32(vdupq_n_f32(127),vdivq_f32(lo,vs)));
                        hi=vmaxq_f32(vdupq_n_f32(-127),vminq_f32(vdupq_n_f32(127),vdivq_f32(hi,vs)));
                        const auto q=vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(lo)),vqmovn_s32(vcvtnq_s32_f32(hi))));
                        const auto q32=vreinterpret_s32_s8(q);
                        const int32_t qlo=vget_lane_s32(q32,0),qhi=vget_lane_s32(q32,1);
                        auto* dest=a+(row/16)*k*16+(col/4)*64+(row%16)*4;
                        std::memcpy(dest,&qlo,4);std::memcpy(dest+64,&qhi,4);
                        vst1_s8(ai+(row/8)*k*8+(col/8)*64+(row%8/2)*16+(row%2)*8,q);
                    }
                }
#endif
                for (;col<end;col+=4) {
                    auto value=row<m ? load_bf16x4(x+row*k+col) : vdupq_n_f32(0);
                    if (inverse) value=vmulq_f32(value,vld1q_f32(inverse+col));
                    value=vmaxq_f32(vdupq_n_f32(-127),vminq_f32(vdupq_n_f32(127),vdivq_f32(value,vs)));
                    const auto q16=vqmovn_s32(vcvtnq_s32_f32(value));
                    const auto q8=vqmovn_s16(vcombine_s16(q16,q16));
                    const int32_t packed=vget_lane_s32(vreinterpret_s32_s8(q8),0);
                    std::memcpy(a+(row/16)*k*16+(col/4)*64+(row%16)*4,&packed,4);
                    if constexpr (HYBRID)
                        std::memcpy(ai+(row/8)*k*8+(col/8)*64+(row%8/2)*16+(row%2)*8+col%8,&packed,4);
                }
            }
#endif
        }
        if (profile) {
#pragma omp master
            { times[0]=omp_get_wtime()-started; }
#pragma omp barrier
        }
        const int64_t count=omp_get_num_threads(), id=omp_get_thread_num();
        if constexpr (HYBRID) {
            if (id<sme_threads) {
                const auto done=split_queue
                    ? compute<true,true,false,true>(a,w,scratch,k,np,mt,0,mt*nt,xs,ws,bias,y,m,n,next_ptr)
                    : compute<true,true>(a,w,scratch,k,np,mt,0,mt*nt,xs,ws,bias,y,m,n,next_ptr);
                if (profile) __atomic_fetch_add(route_counts,done*(split_queue?2:1),__ATOMIC_RELAXED);
            } else {
                int64_t done=0;
                auto claim_neon=[&] {
                    if(split_queue) {
                        auto* queue=reinterpret_cast<uint64_t*>(next_ptr);
                        uint64_t state=__atomic_load_n(queue,__ATOMIC_RELAXED);
                        for(;;) {
                            const uint32_t front=uint32_t(state),back=uint32_t(state>>32);
                            if(front>=back)return mt*nt*2;
                            const uint64_t desired=(uint64_t(back-1)<<32)|front;
                            if(__atomic_compare_exchange_n(queue,&state,desired,true,__ATOMIC_RELAXED,__ATOMIC_RELAXED))
                                return int64_t(back-1);
                        }
                    }
#if defined(FLAGGEMS_W8_NEON_RESERVE)
                    // Short hybrid experiment: leave the final jobs for SME
                    // workers instead of handing a late full tile to Neon.
                    // CAS claims only real jobs; reserved jobs remain visible
                    // to the existing SME fetch-add queue. No math changes.
                    if(m<=64 && sme_threads>0) {
                        const int64_t limit=std::max<int64_t>(0,mt*nt-sme_threads*FLAGGEMS_W8_NEON_RESERVE);
                        int64_t job=__atomic_load_n(next_ptr,__ATOMIC_RELAXED);
                        while(job<limit) {
                            if(__atomic_compare_exchange_n(next_ptr,&job,job+1,true,
                                    __ATOMIC_RELAXED,__ATOMIC_RELAXED))return job;
                        }
                        return mt*nt;
                    }
#endif
                    return __atomic_fetch_add(next_ptr,1,__ATOMIC_RELAXED);
                };
                for (int64_t job=claim_neon();job<mt*nt*(split_queue?2:1);job=claim_neon()) {
                    bool tall=false;
#if defined(FLAGGEMS_W8_M32_TILE)
                    tall=m==32;
#endif
                    const int64_t tile=split_queue?job/2:job;
                    const int64_t row=split_queue?(tile%mt*16+(job%2)*8):(tall?0:tile%mt*16);
                    const int64_t col=tall?tile*32:tile/mt*64;
                    for (int rr=0;rr<(split_queue?8:(tall?32:16));rr+=8) for (int cc=0;cc<(tall?32:64);cc+=8)
                        w8a8_i8mm_tile8(ai+(row+rr)/8*k*8,wi+(col+cc)/8*k*8,
                            xs,ws,bias,y,row+rr,col+cc,m,n,k);
                    ++done;
                }
                if (profile) __atomic_fetch_add(route_counts+1,done,__ATOMIC_RELAXED);
            }
        } else {
            compute<FUSED>(a,w,scratch,k,np,mt,mt*nt*id/count,mt*nt*(id+1)/count,xs,ws,bias,y,m,n);
        }
        if constexpr (NORM) {
#pragma omp barrier
            int64_t norm_descriptor[8]={int64_t(y),d[17],d[18],d[19],m,n,d[20],d[21]};
            flag_gems_arm_add_layer_norm_rows(uint64_t(norm_descriptor));
        }
        if (!FUSED || profile) {
#pragma omp barrier
        }
        if (profile) {
#pragma omp master
            { times[1]=omp_get_wtime()-started-times[0]; }
#pragma omp barrier
        }
        if constexpr (!FUSED) {
#pragma omp for schedule(static)
            for (int64_t row=0; row<m; ++row) for (int64_t col=0; col<n; ++col) {
                float value=(float(scratch[row*np+col])*xs[row])*ws[col];
                if (bias) value+=bf16_float(bias[col]);
                y[row*n+col]=float_bf16(value);
            }
        }
        if (profile) {
#pragma omp master
            { times[2]=omp_get_wtime()-started-times[0]-times[1]; }
        }
    };
    if constexpr(IN_TEAM) {
        worker();
    } else {
#pragma omp parallel num_threads(threads)
        { worker(); }
    }
    if (profile) times[3]=omp_get_wtime()-started;
    if constexpr (HYBRID) if (profile) {times[4]=route_counts[0];times[5]=route_counts[1];}
}

#if !defined(FLAGGEMS_W8_NORM_ONLY) && !defined(FLAGGEMS_W8_PANEL_ONLY) && !defined(FLAGGEMS_W8_FFN_ONLY)
extern "C" void flag_gems_arm_w8a8_sme(uint64_t address) { run(address,false); }
extern "C" void flag_gems_arm_w8a8_sme_scaled(uint64_t address) { run(address,true); }
extern "C" void flag_gems_arm_w8a8_sme_profile(uint64_t address) { run(address,false,true); }
extern "C" void flag_gems_arm_w8a8_sme_scaled_profile(uint64_t address) { run(address,true,true); }
extern "C" void flag_gems_arm_w8a8_sme_fused(uint64_t address) { run<true>(address,false); }
extern "C" void flag_gems_arm_w8a8_sme_scaled_fused(uint64_t address) { run<true>(address,true); }
extern "C" void flag_gems_arm_w8a8_sme_fused_profile(uint64_t address) { run<true>(address,false,true); }
extern "C" void flag_gems_arm_w8a8_sme_scaled_fused_profile(uint64_t address) { run<true>(address,true,true); }
extern "C" void flag_gems_arm_w8a8_sme_hybrid(uint64_t address) { run<true,true>(address,false); }
extern "C" void flag_gems_arm_w8a8_sme_scaled_hybrid(uint64_t address) { run<true,true>(address,true); }
extern "C" void flag_gems_arm_w8a8_sme_hybrid_profile(uint64_t address) { run<true,true>(address,false,true); }
extern "C" void flag_gems_arm_w8a8_sme_scaled_hybrid_profile(uint64_t address) { run<true,true>(address,true,true); }

#endif

#if defined(FLAGGEMS_W8_FFN_ONLY)
// Reuse the existing 44-field FFN descriptor. The table and scalar-tail
// boundaries match the existing GELU leaf; changing the team does not change
// activation rounding. This experimental entry is invoked through Triton Raw.
using GeluBF=c10::BFloat16;
using GeluVec=at::vec::Vectorized<GeluBF>;
static const std::array<GeluBF,65536>& ffn_gelu_table() {
    static const auto table=[] {
        std::array<GeluBF,65536> values;
        alignas(64) uint16_t bits[GeluVec::size()];
        for(int i=0;i<65536;i+=GeluVec::size()) {
            for(int j=0;j<GeluVec::size();++j)bits[j]=uint16_t(i+j);
            at::native::vectorized_gelu<GeluBF>(GeluVec::loadu(bits)).store(values.data()+i);
        }
        return values;
    }();
    return table;
}
static void ffn_gelu_worker(const int64_t* d,const GeluBF* table) {
    const auto* x=reinterpret_cast<const uint16_t*>(d[0]);
    auto* y=reinterpret_cast<GeluBF*>(d[1]);
    const int64_t n=d[2],reference_threads=d[4];
    const int64_t grain=n>16384 ? n/reference_threads : 32768;
    const int64_t tasks=n>grain ? std::min(reference_threads,(n+grain-1)/grain) : 1;
    const int64_t chunk=(n+tasks-1)/tasks;
#if defined(FLAGGEMS_W8_FFN_GELU_SUBTEAM)
    const int64_t workers=std::min<int64_t>(d[3],omp_get_num_threads());
    const int64_t id=omp_get_thread_num();
    for(int64_t task=id<workers?id:tasks;task<tasks;task+=workers) {
#else
#pragma omp for schedule(static)
    for(int64_t task=0;task<tasks;++task) {
#endif
        const auto begin=task*chunk,end=std::min(begin+chunk,n);
        const auto vector_end=begin+(end-begin)/(2*GeluVec::size())*(2*GeluVec::size());
        for(int64_t j=begin;j<vector_end;++j)y[j]=table[x[j]];
        for(int64_t j=vector_end;j<end;++j) {
            GeluBF value;std::memcpy(&value,x+j,2);
            y[j]=at::native::scalar_gelu<GeluBF>(value);
        }
    }
#if defined(FLAGGEMS_W8_FFN_GELU_SUBTEAM)
#pragma omp barrier
#endif
}
#if defined(FLAGGEMS_W8_FFN_GELU_SCALE)
// Preserve the reference GELU chunk boundaries and BF16 rounding, but compute
// each down-projection row scale while its GELU output is still worker-local.
// This removes a separate whole-tensor scale pass and its OpenMP barrier.
static void ffn_gelu_scale_worker(const int64_t* d,const GeluBF* table) {
    auto* x=reinterpret_cast<uint16_t*>(d[17]);
    auto* y=reinterpret_cast<GeluBF*>(d[18]);
    auto* scales=reinterpret_cast<float*>(d[26]);
    const auto* inverse=reinterpret_cast<const float*>(d[34]);
    const int64_t total=d[19],reference_threads=d[21],m=d[30],k=d[32];
    const int64_t grain=total>16384?total/reference_threads:32768;
    const int64_t tasks=total>grain?std::min(reference_threads,(total+grain-1)/grain):1;
    const int64_t chunk=(total+tasks-1)/tasks;
#pragma omp for schedule(static)
    for(int64_t row=0;row<(m+15)/16*16;++row) {
        auto vmax=vdupq_n_f32(0);
        if(row<m) {
            const int64_t end=(row+1)*k;
            for(int64_t pos=row*k;pos<end;) {
                const int64_t begin=(pos/chunk)*chunk;
                const int64_t task_end=std::min(begin+chunk,total);
                const int64_t vector_end=begin+(task_end-begin)/(2*GeluVec::size())*(2*GeluVec::size());
                if(pos<vector_end) {
                    const auto stop=std::min(end,vector_end);
                    for(;pos<stop;++pos)y[pos]=table[x[pos]];
                } else {
                    const auto stop=std::min(end,task_end);
                    for(;pos<stop;++pos) {
                        GeluBF value;std::memcpy(&value,x+pos,2);
                        y[pos]=at::native::scalar_gelu<GeluBF>(value);
                    }
                }
            }
            for(int64_t col=0;col<k;col+=4) {
                auto value=load_bf16x4(reinterpret_cast<const uint16_t*>(y)+row*k+col);
                if(inverse)value=vmulq_f32(value,vld1q_f32(inverse+col));
                vmax=vmaxq_f32(vmax,vabsq_f32(value));
            }
        }
        scales[row]=std::max(vmaxvq_f32(vmax)/127.0f,1e-8f);
    }
}
#endif
extern "C" void flag_gems_arm_w8a8_sme_ffn_composed(uint64_t address) {
    const auto* d=reinterpret_cast<const int64_t*>(address);
    const auto* table=ffn_gelu_table().data();
    run<true,true>(address,d[12]!=0);
    const int threads=int(d[20]);
#pragma omp parallel num_threads(threads)
    { ffn_gelu_worker(d+17,table); }
    run<true,true,true>(uint64_t(d+22),d[34]!=0);
}
extern "C" void flag_gems_arm_w8a8_sme_ffn_team(uint64_t address) {
    const auto* d=reinterpret_cast<const int64_t*>(address);
    // Preserve independently requested teams if they cannot share one team.
    if(d[11]!=d[33] ||
#if defined(FLAGGEMS_W8_FFN_GELU_SUBTEAM)
       d[20]>d[11]
#else
       d[20]!=d[11]
#endif
       ) {
        flag_gems_arm_w8a8_sme_ffn_composed(address);return;
    }
    const auto* table=ffn_gelu_table().data();
    alignas(64) int64_t next_up=0;
    alignas(64) int64_t next_down=0;
    const int threads=int(d[11]);
#pragma omp parallel num_threads(threads)
    {
        run<true,true,false,true>(address,d[12]!=0,false,&next_up);
#pragma omp barrier
#if defined(FLAGGEMS_W8_FFN_GELU_SCALE)
        if(d[8]<=64) {
            ffn_gelu_scale_worker(d,table);
            run<true,true,true,true,true>(uint64_t(d+22),d[34]!=0,false,&next_down);
        } else {
            ffn_gelu_worker(d+17,table);
            run<true,true,true,true>(uint64_t(d+22),d[34]!=0,false,&next_down);
        }
#else
        ffn_gelu_worker(d+17,table);
        run<true,true,true,true>(uint64_t(d+22),d[34]!=0,false,&next_down);
#endif
    }
}
#endif

#if defined(FLAGGEMS_W8_FFN_TENSOR_RAW)
#include "raw_dispatch.h"
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <optional>
#include <thread>
namespace {
using MaybeTensor=std::optional<at::Tensor>;
// Typed entry keeps validation and caller-visible output ownership while moving
// temporary descriptor/scratch preparation out of the Python hot path.
at::Tensor sme_linear_tensor_raw(const at::Tensor& input,const at::Tensor& w,
        const at::Tensor& wi,const at::Tensor& scales,const MaybeTensor& inverse,
        const MaybeTensor& bias,const MaybeTensor& residual,const MaybeTensor& nw,
        const MaybeTensor& nb,int64_t eps_bits,int64_t threads,int64_t sme,int64_t nt) {
    return flag_gems::arm::invoke_raw([&] {
        TORCH_CHECK(input.is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()>=1,
                    "Linear Raw requires CPU BF16 input");
        const int64_t k=input.size(-1),n=scales.numel();
        TORCH_CHECK(k>0 && k%8==0 && n>0,"Invalid linear dimensions");
        auto check=[](const at::Tensor& t,at::ScalarType type) {
            TORCH_CHECK(t.is_cpu() && t.scalar_type()==type && t.is_contiguous(),
                        "Expected contiguous CPU parameters with matching dtypes");
        };
        check(w,at::kChar);check(wi,at::kChar);check(scales,at::kFloat);
        const int64_t padded=(n+63)/64*64;
        TORCH_CHECK(w.dim()==4 && w.size(0)==padded/64 && w.size(1)==k/4
            && w.size(2)==64 && w.size(3)==4,"Invalid SME packed shape");
        TORCH_CHECK(wi.dim()==4 && wi.size(0)==padded/8 && wi.size(1)==k/8
            && wi.size(2)==8 && wi.size(3)==8,"Invalid I8MM packed shape");
        TORCH_CHECK(scales.dim()==1,"Expected per-channel scales");
        if(inverse){check(*inverse,at::kFloat);TORCH_CHECK(inverse->dim()==1 && inverse->numel()==k,"Invalid smoothing shape");}
        if(bias){check(*bias,at::kBFloat16);TORCH_CHECK(bias->dim()==1 && bias->numel()==n,"Invalid bias shape");}
        static const auto limit=std::max<unsigned>(1,std::thread::hardware_concurrency());
        TORCH_CHECK(threads>0 && threads<=limit && sme>=0 && sme<=threads,"Invalid linear worker counts");
        auto shape=input.sizes().vec();shape.back()=n;
        TORCH_CHECK(bool(residual)==bool(nw) && bool(residual)==bool(nb),"Incomplete residual/norm parameters");
        if(residual) {
            check(*residual,at::kBFloat16);check(*nw,at::kBFloat16);check(*nb,at::kBFloat16);
            TORCH_CHECK(input.dim()>=2 && n<=4096 && residual->sizes()==at::IntArrayRef(shape)
                && nw->dim()==1 && nw->numel()==n && nb->sizes()==nw->sizes(),"Invalid residual/norm shapes");
            TORCH_CHECK(nt>0 && nt<=limit,"Invalid norm worker count");
            const uint32_t bits=uint32_t(eps_bits);float eps;std::memcpy(&eps,&bits,4);
            TORCH_CHECK(std::isfinite(eps) && eps>0,"Invalid norm epsilon");
        }
        if(at::GradMode::is_enabled()) {
            TORCH_CHECK(!input.requires_grad() && !scales.requires_grad(),"Linear Raw is inference only");
            for(const auto* t:{&inverse,&bias,&residual,&nw,&nb})
                TORCH_CHECK(!*t || !(*t)->requires_grad(),"Linear Raw is inference only");
        }
        auto output=at::empty(shape,input.options());
        const int64_t m=input.numel()/k;
        if(m) {
            auto x=input.reshape({m,k}).contiguous();
            const int64_t mp=(m+15)/16*16;
            thread_local at::Tensor a,ai,scale;
            auto reserve=[&](at::Tensor& t,int64_t count,at::ScalarType type) {
                if(!t.defined() || t.numel()<count)t=at::empty({count},input.options().dtype(type));
            };
            reserve(a,mp*k,at::kChar);reserve(ai,mp*k,at::kChar);reserve(scale,mp,at::kFloat);
            auto ptr=[](const at::Tensor& t){return reinterpret_cast<int64_t>(t.const_data_ptr());};
            int64_t d[22]={ptr(x),ptr(w),ptr(scales),ptr(a),ptr(scale),0,ptr(output),bias?ptr(*bias):0,
                m,n,k,threads,inverse?ptr(*inverse):0,0,ptr(wi),ptr(ai),sme,
                residual?ptr(*residual):0,nw?ptr(*nw):0,nb?ptr(*nb):0,eps_bits,nt};
            if(residual)run<true,true,true>(reinterpret_cast<uint64_t>(d),bool(inverse));
            else run<true,true>(reinterpret_cast<uint64_t>(d),bool(inverse));
        }
        return output;
    });
}
at::Tensor sme_ffn_tensor_raw(const at::Tensor& input,const at::Tensor& residual,
        const at::Tensor& uw,const at::Tensor& ui8,const at::Tensor& us,
        const MaybeTensor& uinv,const MaybeTensor& ub,
        const at::Tensor& dw,const at::Tensor& di8,const at::Tensor& ds,
        const MaybeTensor& dinv,const MaybeTensor& db,
        const at::Tensor& nw,const at::Tensor& nb,int64_t eps_bits,
        int64_t ut,int64_t ust,int64_t dt,int64_t dst,int64_t gt,int64_t grt,int64_t nt) {
    return flag_gems::arm::invoke_raw([&] {
        TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()>=2,
                    "FFN Raw requires CPU BF16 input with rank >=2");
        const int64_t k=input.size(-1),h=us.numel(),n=ds.numel();
        TORCH_CHECK(k>0 && k%8==0 && h>0 && h%8==0 && n>0 && n<=4096,"Invalid FFN dimensions");
        const int64_t m=input.numel()/k;
#ifndef FLAGGEMS_W8_FFN_MAX_ROWS
#define FLAGGEMS_W8_FFN_MAX_ROWS 32
#endif
        TORCH_CHECK(m<=FLAGGEMS_W8_FFN_MAX_ROWS,"FFN Raw row limit exceeded");
        static const auto limit=std::max<unsigned>(1,std::thread::hardware_concurrency());
        for(auto t:{ut,dt,gt,grt,nt})TORCH_CHECK(t>0 && t<=limit,"Invalid FFN thread count");
        TORCH_CHECK(ust>=0 && ust<=ut && dst>=0 && dst<=dt,"Invalid SME thread count");
        auto check=[](const at::Tensor& t,at::ScalarType type) {
            TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==type && t.is_contiguous(),
                        "FFN Raw requires contiguous CPU parameters with matching dtypes");
        };
        auto check_plan=[&](const at::Tensor& w,const at::Tensor& wi,const at::Tensor& scale,
                const MaybeTensor& inv,const MaybeTensor& bias,int64_t width,int64_t inner) {
            check(w,at::kChar);check(wi,at::kChar);check(scale,at::kFloat);
            const int64_t padded=(width+63)/64*64;
            TORCH_CHECK(w.dim()==4 && w.size(0)==padded/64 && w.size(1)==inner/4
                && w.size(2)==64 && w.size(3)==4,"Invalid SME packed shape");
            TORCH_CHECK(wi.dim()==4 && wi.size(0)==padded/8 && wi.size(1)==inner/8
                && wi.size(2)==8 && wi.size(3)==8,"Invalid I8MM packed shape");
            TORCH_CHECK(scale.dim()==1 && scale.numel()==width,"Invalid FFN scales");
            if(inv){check(*inv,at::kFloat);TORCH_CHECK(inv->dim()==1 && inv->numel()==inner,"Invalid smoothing shape");}
            if(bias){check(*bias,at::kBFloat16);TORCH_CHECK(bias->dim()==1 && bias->numel()==width,"Invalid bias shape");}
        };
        check_plan(uw,ui8,us,uinv,ub,h,k);check_plan(dw,di8,ds,dinv,db,n,h);
        check(residual,at::kBFloat16);check(nw,at::kBFloat16);check(nb,at::kBFloat16);
        auto shape=input.sizes().vec();shape.back()=n;
        TORCH_CHECK(residual.sizes()==at::IntArrayRef(shape) && nw.dim()==1 && nw.numel()==n
            && nb.sizes()==nw.sizes(),"Invalid FFN residual/norm shape");
        const uint32_t bits=uint32_t(eps_bits);float eps;std::memcpy(&eps,&bits,4);
        TORCH_CHECK(std::isfinite(eps) && eps>0,"Invalid FFN epsilon");
        if(at::GradMode::is_enabled()) {
            for(const auto& t:{input,residual,nw,nb})TORCH_CHECK(!t.requires_grad(),"FFN Raw is inference only");
            TORCH_CHECK((!ub || !ub->requires_grad()) && (!db || !db->requires_grad()),"FFN Raw is inference only");
        }
        auto output=at::empty(shape,input.options());
        if(m) {
            auto x=input.reshape({m,k}).contiguous();
            const int64_t mp=(m+15)/16*16,capacity=mp*std::max(k,h);
            thread_local at::Tensor a,ai,scale,hidden;
            auto reserve=[&](at::Tensor& t,int64_t count,at::ScalarType dtype) {
                if(!t.defined() || t.numel()<count)t=at::empty({count},input.options().dtype(dtype));
            };
            reserve(a,capacity,at::kChar);reserve(ai,capacity,at::kChar);
            reserve(scale,mp,at::kFloat);reserve(hidden,m*h,at::kBFloat16);
            auto ptr=[](const at::Tensor& t){return reinterpret_cast<int64_t>(t.const_data_ptr());};
            int64_t d[44]={ptr(x),ptr(uw),ptr(us),ptr(a),ptr(scale),0,ptr(hidden),ub?ptr(*ub):0,
                m,h,k,ut,uinv?ptr(*uinv):0,0,ptr(ui8),ptr(ai),ust,
                ptr(hidden),ptr(hidden),m*h,gt,grt,
                ptr(hidden),ptr(dw),ptr(ds),ptr(a),ptr(scale),0,ptr(output),db?ptr(*db):0,
                m,n,h,dt,dinv?ptr(*dinv):0,0,ptr(di8),ptr(ai),dst,
                ptr(residual),ptr(nw),ptr(nb),eps_bits,nt};
            flag_gems_arm_w8a8_sme_ffn_team(reinterpret_cast<uint64_t>(d));
        }
        return output;
    });
}
}
TORCH_LIBRARY_FRAGMENT(flag_gems_arm_w8a8_sme,m) {
    m.def("linear.raw(Tensor x, Tensor packed, Tensor i8mm, Tensor scales, Tensor? inverse, Tensor? bias, Tensor? residual, Tensor? norm_weight, Tensor? norm_bias, int eps_bits, int threads, int sme_threads, int norm_threads) -> Tensor");
    m.def("ffn.raw(Tensor x, Tensor residual, Tensor up_packed, Tensor up_i8mm, Tensor up_scales, Tensor? up_inverse, Tensor? up_bias, Tensor down_packed, Tensor down_i8mm, Tensor down_scales, Tensor? down_inverse, Tensor? down_bias, Tensor norm_weight, Tensor norm_bias, int eps_bits, int up_threads, int up_sme, int down_threads, int down_sme, int gelu_threads, int gelu_reference_threads, int norm_threads) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_w8a8_sme,CPU,m) {
    m.impl("linear.raw",sme_linear_tensor_raw);
    m.impl("ffn.raw",sme_ffn_tensor_raw);
}
#endif

#ifdef FLAGGEMS_W8_PANEL_ONLY
extern "C" int flag_gems_arm_w8a8_panel_abi_version() { return 1; }
extern "C" void flag_gems_arm_w8a8_sme_panel_hybrid(uint64_t address) { run<true,true>(address,false); }
extern "C" void flag_gems_arm_w8a8_sme_panel_scaled_hybrid(uint64_t address) { run<true,true>(address,true); }
extern "C" void flag_gems_arm_w8a8_sme_panel_hybrid_profile(uint64_t address) { run<true,true>(address,false,true); }
extern "C" void flag_gems_arm_w8a8_sme_panel_scaled_hybrid_profile(uint64_t address) { run<true,true>(address,true,true); }
#endif

#ifdef FLAGGEMS_ARM_FUSED_NORM
extern "C" int flag_gems_arm_w8a8_norm_abi_version() { return 2; }
extern "C" void flag_gems_arm_w8a8_sme_hybrid_norm(uint64_t address) { run<true,true,true>(address,false); }
extern "C" void flag_gems_arm_w8a8_sme_scaled_hybrid_norm(uint64_t address) { run<true,true,true>(address,true); }
#endif
