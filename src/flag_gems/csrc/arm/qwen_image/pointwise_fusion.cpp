#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>

namespace {
uint16x4_t round_bf16(float32x4_t values) {
  auto bits=vreinterpretq_u32_f32(values);
  auto rounded=vaddq_u32(bits,vaddq_u32(vdupq_n_u32(0x7fff),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1))));
  auto result=vshrq_n_u32(rounded,16);
  auto nan=vmvnq_u32(vceqq_f32(values,values));
  return vmovn_u32(vbslq_u32(nan,vdupq_n_u32(0x7fc0),result));
}
uint16x4_t fused_four(const uint16_t* gate,const uint16_t* up,const uint16_t* table) {
  uint16x4_t s={table[gate[0]],table[gate[1]],table[gate[2]],table[gate[3]]};
  auto sf=vreinterpretq_f32_u32(vshll_n_u16(s,16));
  auto uf=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(up),16));
  return round_bf16(vmulq_f32(sf,uf));
}

at::Tensor swiglu(const at::Tensor& gate,const at::Tensor& up,const at::Tensor& table) {
  TORCH_CHECK(gate.device().is_cpu() && up.device().is_cpu() &&
              gate.scalar_type()==at::kBFloat16 && up.scalar_type()==at::kBFloat16 &&
              gate.sizes()==up.sizes(),"Expected matching CPU BF16 gate/up tensors");
  TORCH_CHECK(table.device().is_cpu() && table.scalar_type()==at::kBFloat16 && table.dim()==1 &&
              table.numel()==65536 && table.is_contiguous(),"Expected BF16 SiLU table for all 16-bit encodings");
  auto a=gate.contiguous(),b=up.contiguous();
  auto out=at::empty(gate.sizes(),gate.options());
  const auto* g=reinterpret_cast<const uint16_t*>(a.const_data_ptr<at::BFloat16>());
  const auto* u=reinterpret_cast<const uint16_t*>(b.const_data_ptr<at::BFloat16>());
  const auto* lut=reinterpret_cast<const uint16_t*>(table.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  at::parallel_for(0,gate.numel(),4096,[&](int64_t begin,int64_t end) {
    int64_t i=begin;
    for(;i+4<=end;i+=4) {
      vst1_u16(dst+i,fused_four(g+i,u+i,lut));
    }
    for(;i<end;++i) {
      auto s=at::BFloat16(lut[g[i]],at::BFloat16::from_bits());
      auto v=at::BFloat16(u[i],at::BFloat16::from_bits());
      dst[i]=at::BFloat16(float(s)*float(v)).x;
    }
  });
  return out;
}

at::Tensor swiglu_pack(const at::Tensor& gate,const at::Tensor& up,const at::Tensor& table,int64_t mr) {
  TORCH_CHECK(gate.device().is_cpu() && up.device().is_cpu() &&
              gate.scalar_type()==at::kBFloat16 && up.scalar_type()==at::kBFloat16 &&
              gate.dim()==2 && gate.size(1)>0 && gate.sizes()==up.sizes(),"Expected matching CPU BF16 [M,K]");
  TORCH_CHECK(table.device().is_cpu() && table.scalar_type()==at::kBFloat16 && table.dim()==1 &&
              table.numel()==65536 && table.is_contiguous(),"Expected BF16 SiLU table");
  TORCH_CHECK(mr>0 && mr<=256 && mr%4==0,"Invalid packed panel rows");
  const int64_t m=gate.size(0),k=gate.size(1),kp=(k+1)/2*2,blocks=(m+mr-1)/mr;
  auto a=gate.contiguous(),b=up.contiguous();
  auto out=at::empty({blocks*mr*kp*2},gate.options().dtype(at::kByte));
  const auto* g=reinterpret_cast<const uint16_t*>(a.const_data_ptr<at::BFloat16>());
  const auto* u=reinterpret_cast<const uint16_t*>(b.const_data_ptr<at::BFloat16>());
  const auto* lut=reinterpret_cast<const uint16_t*>(table.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<uint8_t>());
  at::parallel_for(0,blocks,1,[&](int64_t begin,int64_t end) {
    for(int64_t block=begin;block<end;++block) {
      auto* packed=dst+block*mr*kp;
      for(int64_t kb=0;kb<kp;kb+=64) {
        const int64_t ke=std::min(kb+64,kp);
        for(int64_t row=0;row<mr;row+=4) {
          int64_t p=kb;
          if(block*mr+row+4<=m) {
            for(;p+8<=std::min(ke,k);p+=8) {
              uint32x4_t values[4];
              for(int64_t r=0;r<4;++r) {
                const int64_t offset=(block*mr+row+r)*k+p;
                auto lo=fused_four(g+offset,u+offset,lut);
                auto hi=fused_four(g+offset+4,u+offset+4,lut);
                values[r]=vreinterpretq_u32_u16(vcombine_u16(lo,hi));
              }
              auto ab0=vreinterpretq_u64_u32(vtrn1q_u32(values[0],values[1]));
              auto ab1=vreinterpretq_u64_u32(vtrn2q_u32(values[0],values[1]));
              auto cd0=vreinterpretq_u64_u32(vtrn1q_u32(values[2],values[3]));
              auto cd1=vreinterpretq_u64_u32(vtrn2q_u32(values[2],values[3]));
              vst1q_u16(packed+p*mr+2*row,vreinterpretq_u16_u64(vtrn1q_u64(ab0,cd0)));
              vst1q_u16(packed+(p+2)*mr+2*row,vreinterpretq_u16_u64(vtrn1q_u64(ab1,cd1)));
              vst1q_u16(packed+(p+4)*mr+2*row,vreinterpretq_u16_u64(vtrn2q_u64(ab0,cd0)));
              vst1q_u16(packed+(p+6)*mr+2*row,vreinterpretq_u16_u64(vtrn2q_u64(ab1,cd1)));
            }
          }
          for(int64_t r=row;r<row+4;++r) {
            const int64_t index=block*mr+r;
            for(int64_t q=p;q<ke;++q) {
              uint16_t result=0;
              if(index<m && q<k) {
                const int64_t offset=index*k+q;
                auto s=at::BFloat16(lut[g[offset]],at::BFloat16::from_bits());
                auto v=at::BFloat16(u[offset],at::BFloat16::from_bits());
                result=at::BFloat16(float(s)*float(v)).x;
              }
              packed[(q/2)*2*mr+2*r+q%2]=result;
            }
          }
        }
      }
    }
  });
  return out;
}

at::Tensor residual(const at::Tensor& input,const at::Tensor& branch,const at::Tensor& gate) {
  TORCH_CHECK(input.device().is_cpu() && branch.device().is_cpu() && gate.device().is_cpu() &&
              input.scalar_type()==at::kBFloat16 && branch.scalar_type()==at::kBFloat16 && gate.scalar_type()==at::kBFloat16 &&
              input.dim()==3 && input.size(2)>0 && input.sizes()==branch.sizes(),"Expected matching CPU BF16 [B,S,D] inputs");
  TORCH_CHECK(gate.dim()==3 && gate.size(0)==input.size(0) && gate.size(2)==input.size(2) &&
              (gate.size(1)==1 || gate.size(1)==input.size(1)),"Expected per-token or per-batch BF16 gate");
  auto x=input.contiguous(),y=branch.contiguous(),g=gate.contiguous();
  auto out=at::empty(input.sizes(),input.options());
  const auto* xp=reinterpret_cast<const uint16_t*>(x.const_data_ptr<at::BFloat16>());
  const auto* yp=reinterpret_cast<const uint16_t*>(y.const_data_ptr<at::BFloat16>());
  const auto* gp=reinterpret_cast<const uint16_t*>(g.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  const int64_t b=input.size(0),s=input.size(1),d=input.size(2);
  at::parallel_for(0,b*s,std::max<int64_t>(1,4096/d),[&](int64_t begin,int64_t end) {
    for(int64_t row=begin;row<end;++row) {
      const int64_t grow=gate.size(1)==1?row/s:row;
      int64_t col=0;
      for(;col+4<=d;col+=4) {
        auto xf=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(xp+row*d+col),16));
        auto yf=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(yp+row*d+col),16));
        auto gf=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(gp+grow*d+col),16));
        // Preserve the BF16 multiply's rounding before the residual addition.
        auto product=vreinterpretq_f32_u32(vshll_n_u16(round_bf16(vmulq_f32(yf,gf)),16));
        vst1_u16(dst+row*d+col,round_bf16(vaddq_f32(xf,product)));
      }
      for(;col<d;++col) {
        const auto a=at::BFloat16(xp[row*d+col],at::BFloat16::from_bits());
        const auto yv=at::BFloat16(yp[row*d+col],at::BFloat16::from_bits());
        const auto gv=at::BFloat16(gp[grow*d+col],at::BFloat16::from_bits());
        const auto product=at::BFloat16(float(yv)*float(gv));
        dst[row*d+col]=at::BFloat16(float(a)+float(product)).x;
      }
    }
  });
  return out;
}
}
TORCH_LIBRARY(qwen21_pointwise,m) {
  m.def("swiglu(Tensor gate, Tensor up, Tensor table) -> Tensor",&swiglu);
  m.def("swiglu_pack(Tensor gate, Tensor up, Tensor table, int mr) -> Tensor",&swiglu_pack);
  m.def("residual(Tensor input, Tensor branch, Tensor gate) -> Tensor",&residual);
}
