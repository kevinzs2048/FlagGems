#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>

namespace {
uint16x4_t bf16_round(float32x4_t x) {
  auto bits=vreinterpretq_u32_f32(x);
  auto rounded=vaddq_u32(bits,vaddq_u32(vdupq_n_u32(0x7fff),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1))));
  auto nan=vmvnq_u32(vceqq_f32(x,x));
  return vmovn_u32(vbslq_u32(nan,vdupq_n_u32(0x7fc0),vshrq_n_u32(rounded,16)));
}
at::Tensor rope(const at::Tensor& input,const at::Tensor& freqs,int64_t variant) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()==4 &&
              input.size(3)>0 && input.size(3)%2==0,"Expected CPU BF16 [B,S,H,D] with even D");
  const int64_t b=input.size(0),s=input.size(1),h=input.size(2),d=input.size(3);
  TORCH_CHECK(freqs.device().is_cpu() && freqs.scalar_type()==at::kComplexFloat && freqs.dim()==2 &&
              freqs.size(0)==s && freqs.size(1)==d/2,"Expected complex64 [S,D/2] frequencies");
  TORCH_CHECK(variant>=0 && variant<=2,"Invalid multiply schedule");
  auto x=input.contiguous(),f=freqs.contiguous();
  auto out=at::empty(input.sizes(),input.options());
  const auto* src=reinterpret_cast<const uint16_t*>(x.const_data_ptr<at::BFloat16>());
  const auto* coeff=reinterpret_cast<const float*>(f.const_data_ptr<c10::complex<float>>());
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  at::parallel_for(0,b*s*h,32,[&](int64_t begin,int64_t end) {
    for(int64_t row=begin;row<end;++row) {
      const auto* v=src+row*d;
      const auto* c=coeff+((row/h)%s)*d;
      auto* target=dst+row*d;
      int64_t i=0;
      for(;i+8<=d;i+=8) {
        auto packed=vld1q_u16(v+i);
        auto lo=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(packed),16));
        auto hi=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(packed),16));
        auto real=vuzp1q_f32(lo,hi),imag=vuzp2q_f32(lo,hi);
        auto cs=vld2q_f32(c+i);
        float32x4_t r,j;
        if(variant==0) {
          r=vsubq_f32(vmulq_f32(real,cs.val[0]),vmulq_f32(imag,cs.val[1]));
          j=vaddq_f32(vmulq_f32(real,cs.val[1]),vmulq_f32(imag,cs.val[0]));
        } else if(variant==1) {
          r=vfmsq_f32(vmulq_f32(real,cs.val[0]),imag,cs.val[1]);
          j=vfmaq_f32(vmulq_f32(real,cs.val[1]),imag,cs.val[0]);
        } else {
          r=vfmaq_f32(vnegq_f32(vmulq_f32(imag,cs.val[1])),real,cs.val[0]);
          j=vfmaq_f32(vmulq_f32(imag,cs.val[0]),real,cs.val[1]);
        }
        uint16x4x2_t result={{bf16_round(r),bf16_round(j)}};
        vst2_u16(target+i,result);
      }
      for(;i<d;i+=2) {
        float a=float(at::BFloat16(v[i],at::BFloat16::from_bits()));
        float q=float(at::BFloat16(v[i+1],at::BFloat16::from_bits()));
        float r,j;
        if(variant==0) {r=a*c[i]-q*c[i+1];j=a*c[i+1]+q*c[i];}
        else if(variant==1) {r=std::fma(-q,c[i+1],a*c[i]);j=std::fma(q,c[i],a*c[i+1]);}
        else {r=std::fma(a,c[i],-q*c[i+1]);j=std::fma(a,c[i+1],q*c[i]);}
        target[i]=at::BFloat16(r).x;target[i+1]=at::BFloat16(j).x;
      }
    }
  });
  return out;
}
}
TORCH_LIBRARY(qwen21_rope,m) {
  m.def("rope(Tensor input, Tensor freqs, int variant=2) -> Tensor",&rope);
}
