#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <cmath>

namespace {
float32x4_t expand(const uint16_t* ptr) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr),16));
}
uint16x4_t rounded(float32x4_t value) {
  auto bits=vreinterpretq_u32_f32(value);
  auto r=vshrq_n_u32(vaddq_u32(bits,vaddq_u32(vdupq_n_u32(0x7fff),
                    vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1)))),16);
  return vmovn_u32(vbslq_u32(vmvnq_u32(vceqq_f32(value,value)),vdupq_n_u32(0x7fc0),r));
}

// Specialization for the model's contiguous 128-element attention heads.
// Match the installed ATen float32 sum reduction: four vector accumulators,
// then their ordered sum, then an ordered scalar sum of the four lanes.
// Keep square and addition separate, and preserve BF16 rounding BEFORE the
// affine multiplication, as required by the reference Diffusers RMSNorm.
at::Tensor norm_rope128(const at::Tensor& input,const at::Tensor& weight,const at::Tensor& freqs,double eps) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 &&
              input.dim()==4 && input.size(-1)==128,"Expected CPU BF16 input with last dimension 128");
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type()==at::kBFloat16 &&
              weight.dim()==1 && weight.numel()==128,"Expected CPU BF16 weight [128]");
  TORCH_CHECK(freqs.device().is_cpu() && freqs.scalar_type()==at::kComplexFloat &&
              freqs.dim()==2 && freqs.size(0)==input.size(1) && freqs.size(1)==64,
              "Expected complex64 frequencies [S,64]");
  auto x=input.contiguous(),w=weight.contiguous(),f=freqs.contiguous();
  const auto* coeff=reinterpret_cast<const float*>(f.const_data_ptr<c10::complex<float>>());
  const int64_t heads=input.size(2),seq=input.size(1);
  auto output=at::empty(input.sizes(),input.options());
  const auto* src=reinterpret_cast<const uint16_t*>(x.const_data_ptr<at::BFloat16>());
  const auto* affine=reinterpret_cast<const uint16_t*>(w.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(output.data_ptr<at::BFloat16>());
  const float epsilon=static_cast<float>(eps);
  at::parallel_for(0,x.numel()/128,32,[&](int64_t begin,int64_t end) {
    for(int64_t row=begin;row<end;++row) {
      const auto* p=src+row*128;
      float32x4_t accum[4]={vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0)};
      for(int k=0;k<128;k+=16) {
        for(int j=0;j<4;++j) {
          const auto value=expand(p+k+4*j);
          accum[j]=vaddq_f32(accum[j],vmulq_f32(value,value));
        }
      }
      auto sums=vaddq_f32(vaddq_f32(vaddq_f32(accum[0],accum[1]),accum[2]),accum[3]);
      float lanes[4];vst1q_f32(lanes,sums);
      float total=0;
      for(int j=0;j<4;++j)total+=lanes[j];
      const float inverse=1.0f/std::sqrt(total*(1.0f/128.0f)+epsilon);
      const auto* c=coeff+((row/heads)%seq)*128;
      for(int k=0;k<128;k+=8) {
        auto normalize=[&](int offset) {
          auto n=rounded(vmulq_n_f32(expand(p+offset),inverse));
          auto nf=vreinterpretq_f32_u32(vshll_n_u16(n,16));
          auto affine_bf16=rounded(vmulq_f32(nf,expand(affine+offset)));
          return vreinterpretq_f32_u32(vshll_n_u16(affine_bf16,16));
        };
        auto lo=normalize(k),hi=normalize(k+4);
        auto real=vuzp1q_f32(lo,hi),imag=vuzp2q_f32(lo,hi);
        auto cs=vld2q_f32(c+k);
        // Same explicit FMA schedule as the validated standalone CPU RoPE.
        auto r=vfmaq_f32(vnegq_f32(vmulq_f32(imag,cs.val[1])),real,cs.val[0]);
        auto j=vfmaq_f32(vmulq_f32(imag,cs.val[0]),real,cs.val[1]);
        uint16x4x2_t result={{rounded(r),rounded(j)}};
        vst2_u16(dst+row*128+k,result);
      }
    }
  });
  return output;
}
}
TORCH_LIBRARY(qwen21_norm_rope,m) {
  m.def("run(Tensor input, Tensor weight, Tensor freqs, float eps) -> Tensor",&norm_rope128);
}
