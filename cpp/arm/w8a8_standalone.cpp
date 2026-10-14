// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Independent W8A8 channel-scale ABI. No external microkernel library.
#include "raw_dispatch.h"
#include "cpu_features.h"
#include <ATen/Parallel.h>
#include <arm_neon.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>

namespace {
constexpr int NR = 4;
// A block contains K/16 groups of four contiguous 16-byte weight vectors,
// followed by four int32 sums, four FP32 scales and four reserved FP32 zeros.
at::Tensor pack(const at::Tensor& weight, const at::Tensor& scales, bool sme) {
  TORCH_CHECK(!sme, "Standalone W8A8 currently supports Neon only");
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type()==at::kChar &&
    weight.dim()==2 && weight.is_contiguous(), "Expected contiguous CPU INT8 [N,K]");
  auto n=weight.size(0), k=weight.size(1);
  TORCH_CHECK(n>0 && k>0 && k%32==0 && n%4==0, "Expected N%4=0, K%32=0");
  TORCH_CHECK(scales.device().is_cpu() && scales.scalar_type()==at::kFloat &&
    scales.numel()==n && scales.is_contiguous(), "Expected FP32 channel scales");
  auto out=at::empty({n/4*(4*k+48)}, weight.options().dtype(at::kByte));
  auto* dst=out.data_ptr<uint8_t>();auto* w=weight.const_data_ptr<int8_t>();
  auto* scale=scales.const_data_ptr<float>();
  at::parallel_for(0,n/4,16,[&](int64_t begin,int64_t end){
    for(auto b=begin;b<end;++b){
      auto* p=dst+b*(4*k+48);int32_t sums[4]={};
      for(int c=0;c<4;++c){
        int32x4_t sum=vdupq_n_s32(0);
        for(int64_t j=0;j<k;j+=16){
          auto v=vld1q_s8(w+(b*4+c)*k+j);
          vst1q_s8(reinterpret_cast<int8_t*>(p)+j*4+c*16,v);
          sum=vaddq_s32(sum,vpaddlq_s16(vpaddlq_s8(v)));
        }
        sums[c]=vaddvq_s32(sum);
      }
      std::memcpy(p+4*k,sums,16);
      std::memcpy(p+4*k+16,scale+b*4,16);
      std::memset(p+4*k+32,0,16);
    }
  });return out;
}

void quantize_row(const float* input, int8_t* output, int64_t k,
                  int32_t& zero, float& scale) {
  auto lo=vdupq_n_f32(0), hi=vdupq_n_f32(0);
  for(int64_t j=0;j<k;j+=4){auto x=vld1q_f32(input+j);lo=vminq_f32(lo,x);hi=vmaxq_f32(hi,x);}
  float min=vminvq_f32(lo),max=vmaxvq_f32(hi);
  float multiplier=min==max?1.f:255.f/(max-min);
  scale=multiplier?1.f/multiplier:0.f;
  float a=min*multiplier,b=max*multiplier;
  float zp=(-128.f+a)+(127.f+b)>0?-128.f-a:127.f-b;
  zero=int32_t(std::nearbyint(std::clamp(zp,-128.f,127.f)));
  for(int64_t j=0;j<k;j+=16){
    int16x8_t q0=vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(input+j),multiplier))),
                             vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(input+j+4),multiplier))));
    int16x8_t q1=vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(input+j+8),multiplier))),
                             vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(input+j+12),multiplier))));
    auto z=vdupq_n_s16(int16_t(zero));
    vst1q_s8(output+j,vcombine_s8(vqmovn_s16(vaddq_s16(q0,z)),vqmovn_s16(vaddq_s16(q1,z))));
  }
}
struct Plan {
  const float* input;const uint8_t* packed;int8_t* quantized;
  int32_t* zeros;float* scales;float* output;
  int64_t m,n,k;std::exception_ptr error;
};
template<int ROWS>
void tile(const Plan& p,int64_t row,int64_t block) {
  auto* w=p.packed+block*(4*p.k+48);
  int32x4_t accum[ROWS][4];
  for(int r=0;r<ROWS;++r)for(int c=0;c<4;++c)accum[r][c]=vdupq_n_s32(0);
  for(int64_t j=0;j<p.k;j+=16){
    int8x16_t weights[4];for(int c=0;c<4;++c)weights[c]=vld1q_s8(reinterpret_cast<const int8_t*>(w)+j*4+c*16);
    for(int r=0;r<ROWS;++r){
      auto x=vld1q_s8(p.quantized+(row+r)*p.k+j);
      for(int c=0;c<4;++c)accum[r][c]=vdotq_s32(accum[r][c],weights[c],x);
    }
  }
  int32_t sums[4];float scales[4];std::memcpy(sums,w+4*p.k,16);std::memcpy(scales,w+4*p.k+16,16);
  for(int r=0;r<ROWS;++r)for(int c=0;c<4;++c){
    auto dot=vaddvq_s32(accum[r][c])-p.zeros[row+r]*sums[c];
    p.output[(row+r)*p.n+block*4+c]=float(dot)*(p.scales[row+r]*scales[c]);
  }
}
#if defined(__clang__)
__attribute__((target("i8mm")))
#else
__attribute__((target("arch=armv8.2-a+dotprod+i8mm")))
#endif
void tile_i8mm4(const Plan& p,int64_t row,int64_t block) {
  auto* w=p.packed+block*(4*p.k+48);
  int32x4_t acc[2][2]={{vdupq_n_s32(0),vdupq_n_s32(0)},{vdupq_n_s32(0),vdupq_n_s32(0)}};
  for(int64_t j=0;j<p.k;j+=16){
    int8x16_t weights[4];for(int c=0;c<4;++c)weights[c]=vld1q_s8(reinterpret_cast<const int8_t*>(w)+j*4+c*16);
    for(int r=0;r<2;++r){
      auto a0=vld1q_s8(p.quantized+(row+2*r)*p.k+j);
      auto a1=vld1q_s8(p.quantized+(row+2*r+1)*p.k+j);
      auto lo=vcombine_s8(vget_low_s8(a0),vget_low_s8(a1));
      auto hi=vcombine_s8(vget_high_s8(a0),vget_high_s8(a1));
      for(int c=0;c<2;++c){
        auto b0=vcombine_s8(vget_low_s8(weights[2*c]),vget_low_s8(weights[2*c+1]));
        auto b1=vcombine_s8(vget_high_s8(weights[2*c]),vget_high_s8(weights[2*c+1]));
        acc[r][c]=vmmlaq_s32(acc[r][c],lo,b0);acc[r][c]=vmmlaq_s32(acc[r][c],hi,b1);
      }
    }
  }
  int32_t sums[4];float scales[4];std::memcpy(sums,w+4*p.k,16);std::memcpy(scales,w+4*p.k+16,16);
  for(int r=0;r<2;++r)for(int c=0;c<2;++c){
    int32_t result[4];vst1q_s32(result,acc[r][c]);
    for(int dr=0;dr<2;++dr)for(int dc=0;dc<2;++dc){
      auto rr=row+2*r+dr;auto cc=2*c+dc;
      auto dot=result[dr*2+dc]-p.zeros[rr]*sums[cc];
      p.output[rr*p.n+block*4+cc]=float(dot)*(p.scales[rr]*scales[cc]);
    }
  }
}
} // namespace
extern "C" void flag_gems_arm_w8_compute(uint64_t descriptor) noexcept {
  auto& p=*reinterpret_cast<Plan*>(descriptor);
  try {
    at::parallel_for(0,p.m,1,[&](int64_t begin,int64_t end){
      for(auto r=begin;r<end;++r)quantize_row(p.input+r*p.k,p.quantized+r*p.k,p.k,p.zeros[r],p.scales[r]);
    });
    auto blocks=p.n/4, stripes=(blocks+31)/32, row_tiles=(p.m+3)/4;
    const char* experiment=std::getenv("FLAGGEMS_STANDALONE_PREFILL_I8MM");
    bool i8mm=experiment && std::strcmp(experiment,"1")==0 &&
      flag_gems::arm::has_feature(flag_gems::arm::Feature::I8mm);
    at::parallel_for(0,row_tiles*stripes,1,[&](int64_t begin,int64_t end){
      for(auto job=begin;job<end;++job){
        auto row=job/stripes*4,first=job%stripes*32,last=std::min(first+32,blocks);
        for(auto block=first;block<last;++block){
          if(row+4<=p.m && i8mm)tile_i8mm4(p,row,block);
          else if(row+4<=p.m)tile<4>(p,row,block);
          else for(auto r=row;r<p.m;++r)tile<1>(p,r,block);
        }
      }
    });
  } catch(...) {p.error=std::current_exception();}
}
namespace {
at::Tensor linear_impl(const at::Tensor& input,const at::Tensor& packed,int64_t n,
                       bool sme,int64_t tile_rows,int64_t tile_cols,bool raw) {
  TORCH_CHECK(!sme,"Standalone W8A8 currently supports Neon only");
  TORCH_CHECK(flag_gems::arm::has_feature(flag_gems::arm::Feature::DotProd),"Standalone W8A8 requires Arm DotProd");
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kFloat &&
    input.dim()==2 && input.is_contiguous(),"Expected contiguous FP32 CPU [M,K]");
  auto m=input.size(0),k=input.size(1);
  TORCH_CHECK(m>0 && n>0 && n%4==0 && k>0 && k%32==0 && tile_rows>0 && tile_cols>0,"Invalid shape");
  TORCH_CHECK(packed.device().is_cpu() && packed.scalar_type()==at::kByte &&
    packed.is_contiguous() && packed.numel()==n/4*(4*k+48),"Invalid standalone W8 packed weight");
  auto q=at::empty({m,k},packed.options().dtype(at::kChar));
  auto z=at::empty({m},packed.options().dtype(at::kInt));
  auto scale=at::empty({m},input.options());auto output=at::empty({m,n},input.options());
  Plan plan{input.const_data_ptr<float>(),packed.const_data_ptr<uint8_t>(),q.data_ptr<int8_t>(),
            z.data_ptr<int32_t>(),scale.data_ptr<float>(),output.data_ptr<float>(),m,n,k,{}};
  if(raw)flag_gems::arm::invoke_dense_compute(plan,1);
  else flag_gems_arm_w8_compute(reinterpret_cast<uint64_t>(&plan));
  if(plan.error)std::rethrow_exception(plan.error);return output;
}
at::Tensor linear(const at::Tensor& x,const at::Tensor& w,int64_t n,bool sme,int64_t mr,int64_t nr){return linear_impl(x,w,n,sme,mr,nr,false);}
at::Tensor linear_raw(const at::Tensor& x,const at::Tensor& w,int64_t n,bool sme,int64_t mr,int64_t nr){return linear_impl(x,w,n,sme,mr,nr,true);}
}
TORCH_LIBRARY(flag_gems_arm_w8a8,m){
  m.def("implementation() -> str", [](){return std::string("standalone-neon");});
  m.def("pack(Tensor weight, Tensor scales, bool sme) -> Tensor");
  m.def("linear(Tensor input, Tensor packed, int n, bool sme, int tile_rows, int tile_cols) -> Tensor");
  m.def("linear.raw(Tensor input, Tensor packed, int n, bool sme, int tile_rows, int tile_cols) -> Tensor");
}
TORCH_LIBRARY_IMPL(flag_gems_arm_w8a8,CPU,m){m.impl("pack",pack);m.impl("linear",linear);m.impl("linear.raw",linear_raw);}
