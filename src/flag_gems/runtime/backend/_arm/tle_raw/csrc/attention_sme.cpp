// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Native SME2 leaf for CPU Triton TLE Raw. BF16 vision attention with
// experimental packed FP32 probability schedule; quality validation pending.
#include <arm_sme.h>
#include <arm_neon.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>
#include <type_traits>
#include <omp.h>
using BF=at::BFloat16;
#if defined(FLAGGEMS_ATTN_BFMOPA)
using Packed=BF;
#else
using Packed=float;
#endif
#if defined(FLAGGEMS_ATTN_NATIVE_SOFTMAX)
// Diagnostic reproduction of the reduced-precision Arm exp used by vLLM.
static float32x4_t encoder_exp(float32x4_t x) {
    const auto n=vrndaq_f32(vmulq_n_f32(x,0x1.715476p+0f));
    const auto r=vfmsq_n_f32(x,n,0x1.62e43p-1f);
    const auto e=vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)),23);
    const auto q=vfmaq_n_f32(vdupq_n_f32(0x1.017d34p-1f),r,0x1.5592ecp-3f);
    const auto p=vfmaq_f32(vaddq_f32(vdupq_n_f32(1.f),r),q,vmulq_f32(r,r));
    auto y=vreinterpretq_f32_u32(vaddq_u32(vreinterpretq_u32_f32(p),e));
    y=vbslq_f32(vcltq_f32(x,vdupq_n_f32(-0x1.5ebb82p+6f)),vdupq_n_f32(0.f),y);
    return vbslq_f32(vcgtq_f32(x,vdupq_n_f32(0x1.61814ap+6f)),vdupq_n_f32(INFINITY),y);
}
#endif
namespace {
using CPUEntry=void(*)(void*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
std::atomic<CPUEntry> attention_raw_entry{nullptr};
}
extern "C" void flag_gems_arm_bf16_attention_register_raw(uint64_t address) {
    attention_raw_entry.store(reinterpret_cast<CPUEntry>(address),std::memory_order_release);
}
extern "C" void flag_gems_arm_bf16_attention_raw(uint64_t descriptor) {
    auto fn=attention_raw_entry.load(std::memory_order_acquire);
    if(!fn) __builtin_trap();
    fn(reinterpret_cast<void*>(descriptor),0,0,0,1,1,1);
}
// FP32-output BF16 CPU GEMM uses four independent sums in this Torch
// build, unlike its BF16-output Linear dot. Preserve that separate schedule.
__arm_new("za") __arm_locally_streaming
static void compute_sum4(const Packed* a,const Packed* b,float* out,float* partial,
                         int64_t k,int64_t kp,int64_t np,int64_t mt,int64_t begin,int64_t end) {
    if(svcntsw()!=16) __builtin_trap();
    const auto pg=svptrue_b32();
#if defined(FLAGGEMS_ATTN_BFMOPA)
    const auto ph=svptrue_b16();
    for(int64_t job=begin;job<end;++job) {
        const int64_t mi=job%mt,ni=job/mt;
        const auto* ap=reinterpret_cast<const bfloat16_t*>(a+mi*kp*16);
        const auto* bp=reinterpret_cast<const bfloat16_t*>(b+ni*kp*64);
        svzero_za();
        for(int64_t step=0;step<k;step+=2) {
            const auto av=svld1_bf16(ph,ap+(step/2)*32);
            const auto* p=bp+(step/2)*128;
            svmopa_za32_bf16_m(0,ph,ph,av,svld1_bf16(ph,p));
            svmopa_za32_bf16_m(1,ph,ph,av,svld1_bf16(ph,p+32));
            svmopa_za32_bf16_m(2,ph,ph,av,svld1_bf16(ph,p+64));
            svmopa_za32_bf16_m(3,ph,ph,av,svld1_bf16(ph,p+96));
        }
#define SAVE_BF_TILE(T,C) \
        for(int row=0;row<16;row+=4) { \
            const auto four=svread_hor_za32_f32_vg4(T,row); \
            auto* dst=out+(mi*16+row)*np+ni*64+C; \
            svst1_f32(pg,dst,svget4_f32(four,0)); \
            svst1_f32(pg,dst+np,svget4_f32(four,1)); \
            svst1_f32(pg,dst+np*2,svget4_f32(four,2)); \
            svst1_f32(pg,dst+np*3,svget4_f32(four,3)); \
        }
        SAVE_BF_TILE(0,0) SAVE_BF_TILE(1,16) SAVE_BF_TILE(2,32) SAVE_BF_TILE(3,48)
#undef SAVE_BF_TILE
    }
#else
    for(int64_t job=begin;job<end;++job) {
        const int64_t mi=job%mt,ni=job/mt;
        const auto* ap=a+mi*kp*16;const auto* bp=b+ni*kp*64;
        for(int phase=0;phase<4;++phase) {
            svzero_za();
            const auto main=k/4*4;
            for(int64_t step=phase;step<main;step+=4) {
                const auto av=svld1_f32(pg,ap+step*16);const auto* p=bp+step*64;
                svmopa_za32_f32_m(0,pg,pg,av,svld1_f32(pg,p));
                svmopa_za32_f32_m(1,pg,pg,av,svld1_f32(pg,p+16));
                svmopa_za32_f32_m(2,pg,pg,av,svld1_f32(pg,p+32));
                svmopa_za32_f32_m(3,pg,pg,av,svld1_f32(pg,p+48));
            }
            if(phase==0) for(int64_t step=main;step<k;++step) {
                const auto av=svld1_f32(pg,ap+step*16);const auto* p=bp+step*64;
                svmopa_za32_f32_m(0,pg,pg,av,svld1_f32(pg,p));
                svmopa_za32_f32_m(1,pg,pg,av,svld1_f32(pg,p+16));
                svmopa_za32_f32_m(2,pg,pg,av,svld1_f32(pg,p+32));
                svmopa_za32_f32_m(3,pg,pg,av,svld1_f32(pg,p+48));
            }
#define SAVE_SUM4(T,C) \
            for(int row=0;row<16;row+=4) { \
                const auto four=svread_hor_za32_f32_vg4(T,row); \
                auto* dst=partial+phase*1024+row*64+C; \
                svst1_f32(pg,dst,svget4_f32(four,0)); \
                svst1_f32(pg,dst+64,svget4_f32(four,1)); \
                svst1_f32(pg,dst+128,svget4_f32(four,2)); \
                svst1_f32(pg,dst+192,svget4_f32(four,3)); \
            }
            SAVE_SUM4(0,0) SAVE_SUM4(1,16) SAVE_SUM4(2,32) SAVE_SUM4(3,48)
#undef SAVE_SUM4
        }
        for(int row=0;row<16;++row) for(int col=0;col<64;col+=16) {
            const auto offset=row*64+col;
            auto value=svld1_f32(pg,partial+offset);
            for(int phase=1;phase<4;++phase) value=svadd_f32_x(pg,value,svld1_f32(pg,partial+phase*1024+offset));
            svst1_f32(pg,out+(mi*16+row)*np+ni*64+col,value);
        }
    }
#endif
}

// Contiguous output [B,H,Q,D]; Q/K/V may have different strides and GQA.
// Experimental CPU encoder attention: FP32 softmax weights and state.
// BF16 Q/K/V and output; no intermediate probability quantization.
extern "C" void flag_gems_arm_bf16_attention(uint64_t address) {
    const auto* d=reinterpret_cast<const int64_t*>(address);
    const auto* q=reinterpret_cast<const BF*>(d[0]);
    const auto* k=reinterpret_cast<const BF*>(d[1]);
    const auto* v=reinterpret_cast<const BF*>(d[2]);
    auto* out=reinterpret_cast<BF*>(d[3]);
    const int64_t batches=d[4],heads=d[5],kvheads=d[6],qlen=d[7],kvlen=d[8],dim=d[9];
    if(dim!=64) __builtin_trap(); // Experimental BGE specialization.
    const int threads=d[19];const bool causal=d[20];
    float scale;uint32_t scale_bits=uint32_t(d[21]);std::memcpy(&scale,&scale_bits,4);
#if defined(FLAGGEMS_ATTN_FULL_K)
#if defined(FLAGGEMS_ATTN_QM)
    const int64_t qm=FLAGGEMS_ATTN_QM;
#else
    const int64_t qm=64;
#endif
#else
    const int64_t qm=qlen>=768 ? 256 : qlen>=192 ? 64 : 32;
#endif
    const int64_t tiles=(qlen+qm-1)/qm, jobs=batches*heads*tiles;
    using Vec=at::vec::Vectorized<float>;
    // Do not allocate/clear512-key panels for short encoder sequences.
#if defined(FLAGGEMS_ATTN_FULL_K)
    const int64_t kb=(kvlen+63)/64*64;
#else
    const int64_t kb=std::min(int64_t(512),(kvlen+63)/64*64);
#endif
    const int64_t blocks=(kvlen+kb-1)/kb;
    const int64_t head_elements=blocks*kb*dim;
    std::vector<Packed> packed_k(batches*kvheads*head_elements);
    std::vector<Packed> packed_v(batches*kvheads*head_elements);
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t bh=0;bh<batches*kvheads;++bh) {
        const BF* kp=k+(bh/kvheads)*d[13]+(bh%kvheads)*d[14];
        const BF* vp=v+(bh/kvheads)*d[16]+(bh%kvheads)*d[17];
        for(int64_t start=0;start<kvlen;start+=kb) {
            auto* pk=packed_k.data()+bh*head_elements+(start/kb)*kb*dim;
            auto* pv=packed_v.data()+bh*head_elements+(start/kb)*kb*dim;
            const int64_t n=std::min(int64_t(kb),kvlen-start);
            for(int64_t i=0;i<n;++i) for(int64_t j=0;j<dim;++j) {
#if defined(FLAGGEMS_ATTN_BFMOPA)
                pk[(i/64)*dim*64+(j/2)*128+(i%64)*2+j%2]=kp[(start+i)*d[15]+j];
                pv[(i/2)*128+j*2+i%2]=vp[(start+i)*d[18]+j];
#else
                pk[(i/64)*dim*64+j*64+i%64]=float(kp[(start+i)*d[15]+j]);
                pv[i*64+j]=float(vp[(start+i)*d[18]+j]);
#endif
            }
        }
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t job=0;job<jobs;++job) {
        const auto batch=job/(heads*tiles), head=(job/tiles)%heads, row=(job%tiles)*qm;
        const int64_t m=std::min(qm,qlen-row),kh=head/(heads/kvheads);
        const BF* qp=q+batch*d[10]+head*d[11]+row*d[12];
        const BF* kp=k+batch*d[13]+kh*d[14];
        const BF* vp=v+batch*d[16]+kh*d[17];
        const int64_t mp=(m+15)/16*16,mt=mp/16;
        std::vector<float> score(mp*kb),result(mp*dim),next(mp*dim);
        std::vector<Packed> packed_q(mp*dim),packed_p(mp*kb);
        std::vector<float> partial(4096);
        for(int64_t i=0;i<m;++i) for(int64_t j=0;j<dim;++j)
#if defined(FLAGGEMS_ATTN_BFMOPA)
            packed_q[(i/16)*dim*16+(j/2)*32+(i%16)*2+j%2]=BF(float(qp[i*d[12]+j])*scale);
#else
            packed_q[(i/16)*dim*16+j*16+i%16]=float(qp[i*d[12]+j]);
#endif
        std::vector<float> maxima(m,-std::numeric_limits<float>::infinity()),sums(m,0.f);

        const int64_t end=causal ? std::min(kvlen,row+m) : kvlen;
        for(int64_t start=0;start<end;start+=kb) {
            const int64_t n=std::min(int64_t(kb),kvlen-start);
            const int64_t np=(n+63)/64*64;
            const auto off=(batch*kvheads+kh)*head_elements+(start/kb)*kb*dim;
            compute_sum4(packed_q.data(),packed_k.data()+off,score.data(),partial.data(),dim,dim,np,mt,0,mt*(np/64));
            for(int64_t r=0;r<m;++r) {
                auto* scores=score.data()+r*np;
#if defined(FLAGGEMS_ATTN_SUM16)
                // Native attention masks its aligned key tail before the
                // vector exp/reduction, rather than using scalar exp there.
                for(int64_t c=n;c<np;++c)scores[c]=-std::numeric_limits<float>::infinity();
                const int64_t soft_n=np;
#else
                const int64_t soft_n=n;
#endif
                if(causal) for(int64_t c=std::max(int64_t(0),row+r-start+1);c<n;++c)
                    scores[c]=-std::numeric_limits<float>::infinity();
                auto maximum=Vec(-std::numeric_limits<float>::infinity());
#if defined(FLAGGEMS_ATTN_BFMOPA)
                const auto scaling=Vec(1.f);
#else
                const auto scaling=Vec(scale);
#endif
                int64_t c=0;
                for(;c+Vec::size()<=soft_n;c+=Vec::size()) {
                    const auto value=Vec::loadu(scores+c)*scaling;
                    maximum=at::vec::maximum(maximum,value);value.store(scores+c);
                }
                float max_value=at::vec::vec_reduce_all<float>([](const Vec& a,const Vec& b){return at::vec::maximum(a,b);},maximum);
                for(;c<soft_n;++c) {
#if !defined(FLAGGEMS_ATTN_BFMOPA)
                    scores[c]=scores[c]*scale;
#endif
                    max_value=std::max(max_value,scores[c]);
                }
                max_value=std::max(maxima[r],max_value);
#if defined(FLAGGEMS_ATTN_NATIVE_SOFTMAX)
                if (maxima[r]-max_value>=-8.f) max_value=maxima[r];
#endif
                auto sum_vector=Vec(0.f);const auto max_vector=Vec(max_value);c=0;
#if defined(FLAGGEMS_ATTN_SUM16)
                Vec sum16[4]={Vec(0.f),Vec(0.f),Vec(0.f),Vec(0.f)};
#if defined(FLAGGEMS_ATTN_NATIVE_SOFTMAX)
                for(;c+16<=soft_n;c+=16) {
                    for(int lane=0;lane<4;++lane) {
                        const int64_t col=c+lane*4;
                        const auto ex=Vec(encoder_exp(vsubq_f32(vld1q_f32(scores+col),vdupq_n_f32(max_value))));
                        sum16[lane]=sum16[lane]+ex;ex.store(scores+col);
                    }
                }
#endif
#endif
                for(;c+Vec::size()<=soft_n;c+=Vec::size()) {
#if defined(FLAGGEMS_ATTN_NATIVE_SOFTMAX)
                    const auto ex=Vec(encoder_exp(vsubq_f32(vld1q_f32(scores+c),vdupq_n_f32(max_value))));
#else
                    const auto ex=(Vec::loadu(scores+c)-max_vector).fexp_u20();
#endif
#if defined(FLAGGEMS_ATTN_SUM16)
                    const int lane=(c/4)%4;
                    sum16[lane]=sum16[lane]+ex;
#else
                    sum_vector=sum_vector+ex;
#endif
                    ex.store(scores+c);
                }
#if defined(FLAGGEMS_ATTN_SUM16)
                float total=0;
                // std::plus selects Arm's pairwise FADDP reduction, matching
                // the native FP32Vec16 implementation. A lambda instead uses
                // (lane0+lane2)+(lane1+lane3), changing BF16 rounding.
                for(int i=0;i<4;++i)total+=at::vec::vec_reduce_all<float>(std::plus<Vec>{},sum16[i]);
#else
                float total=at::vec::vec_reduce_all<float>([](const Vec& a,const Vec& b){return a+b;},sum_vector);
#endif
                for(;c<soft_n;++c) {scores[c]=std::exp(scores[c]-max_value);total+=scores[c];}
                for(c=0;c<n;++c)
#if defined(FLAGGEMS_ATTN_BFMOPA)
                    packed_p[(r/16)*kb*16+(c/2)*32+(r%16)*2+c%2]=BF(scores[c]);
#elif defined(FLAGGEMS_ATTN_NATIVE_PROB)
                    packed_p[(r/16)*kb*16+c*16+r%16]=float(BF(scores[c]));
#else
                    packed_p[(r/16)*kb*16+c*16+r%16]=scores[c];
#endif
                const float rescale=std::exp(maxima[r]-max_value);
#if defined(FLAGGEMS_ATTN_NATIVE_SOFTMAX)
                sums[r]=rescale*sums[r]+total;maxima[r]=max_value;
#else
                sums[r]=std::fma(rescale,sums[r],total);maxima[r]=max_value;
#endif
                if(start) for(c=0;c<dim;++c) result[r*dim+c]=result[r*dim+c]*rescale;
            }
            compute_sum4(packed_p.data(),packed_v.data()+off,next.data(),partial.data(),n,kb,64,mt,0,mt);
            for(int64_t i=0;i<m*dim;++i) result[i]=start ? result[i]+next[i] : next[i];
        }
        for(int64_t r=0;r<m;++r) {
            const float reciprocal=1.f/sums[r];
            for(int64_t c=0;c<dim;++c) out[((batch*heads+head)*qlen+row+r)*dim+c]=BF(result[r*dim+c]*reciprocal);
        }
    }
}
