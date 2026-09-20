// Experimental: two complete-K W8 projections -> exact rounded SwiGLU ->
// BF16 packed LHS, with only worker-local FP32 projection scratch.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <atomic>
#include <cfloat>
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#define WG(name) kai_##name##_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa
#define WP(name) kai_##name##_rhs_pack_nxk_qsi8cxp_qsi8cx_neon
#define AP(name) kai_##name##_lhs_quant_pack_qai8dxp_f32
#define BG(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa

namespace {
int64_t divup(int64_t x,int64_t y) {return (x+y-1)/y;}
uint16x4_t rounded(float32x4_t value) {
  auto bits=vreinterpretq_u32_f32(value);
  auto r=vshrq_n_u32(vaddq_u32(bits,vaddq_u32(vdupq_n_u32(0x7fff),
                    vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1)))),16);
  return vmovn_u32(vbslq_u32(vmvnq_u32(vceqq_f32(value,value)),vdupq_n_u32(0x7fc0),r));
}
uint16x4_t fused4(const float* gate,const float* up,const uint16_t* table) {
  uint16_t g[4];vst1_u16(g,rounded(vld1q_f32(gate)));
  uint16x4_t silu={table[g[0]],table[g[1]],table[g[2]],table[g[3]]};
  auto sf=vreinterpretq_f32_u32(vshll_n_u16(silu,16));
  auto uf=vreinterpretq_f32_u32(vshll_n_u16(rounded(vld1q_f32(up)),16));
  return rounded(vmulq_f32(sf,uf));
}
at::Tensor run(const at::Tensor& lhs,const at::Tensor& gate_rhs,const at::Tensor& up_rhs,
               const at::Tensor& table,int64_t m,int64_t n,int64_t k,int64_t workers,int64_t mt,int64_t nt) {
  TORCH_CHECK(m>=0 && n>0 && k>0 && k%8==0 && k<=32768 && workers>0 && workers<=at::get_num_threads(),
              "Invalid dimensions/workers");
  const int64_t mr=BG(get_mr)(),kp=divup(n,2)*2;
  TORCH_CHECK(mt>0 && mt<=256 && nt>0 && nt<=2048 && mt%mr==0 && mt%WG(get_m_step)()==0 &&
              nt%WG(get_n_step)()==0,"Invalid tile shape");
  TORCH_CHECK(lhs.device().is_cpu() && lhs.scalar_type()==at::kByte && lhs.dim()==1 && lhs.is_contiguous() &&
      lhs.numel()==AP(get_lhs_packed_size)(m,k,WG(get_mr)(),WG(get_kr)(),WG(get_sr)()),"Invalid packed activation");
  for(const auto& rhs:{gate_rhs,up_rhs})
    TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
      rhs.numel()==WP(get_rhs_packed_size)(n,k,WG(get_nr)(),WG(get_kr)(),WG(get_sr)()),"Invalid packed weight");
  TORCH_CHECK(table.device().is_cpu() && table.scalar_type()==at::kBFloat16 && table.dim()==1 &&
              table.numel()==65536 && table.is_contiguous(),"Invalid BF16 SiLU table");
  auto out=at::empty({divup(m,mr)*mr*kp*2},lhs.options());
  if(!m)return out;
  auto scratch=at::empty({workers,2,mt,nt},lhs.options().dtype(at::kFloat));
  const auto* lut=reinterpret_cast<const uint16_t*>(table.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<uint8_t>());
  const int64_t rows=divup(m,mt),cols=divup(n,nt),padded_m=divup(m,mr)*mr;
  std::atomic<int64_t> next{0};
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      auto* gate=scratch.data_ptr<float>()+slot*2*mt*nt;auto* up=gate+mt*nt;
      int64_t task;
      while((task=next.fetch_add(1,std::memory_order_relaxed))<rows*cols) {
        const int64_t row=task%rows*mt,col=task/rows*nt;
        const int64_t height=std::min(mt,m-row),width=std::min(nt,n-col);
        const auto* x=lhs.const_data_ptr<uint8_t>()+WG(get_lhs_packed_offset)(row,k);
        for(int branch=0;branch<2;++branch) {
          const auto& w=branch?up_rhs:gate_rhs;
          WG(run)(height,width,k,x,w.const_data_ptr<uint8_t>()+WG(get_rhs_packed_offset)(col,k),
                  branch?up:gate,nt*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
        }
        // Each task owns aligned M panels and an even output-column range.
        // Write padding explicitly; workers never share output elements.
        const int64_t padded_height=std::min(mt,padded_m-row),padded_width=std::min(nt,kp-col);
        for(int64_t rb=0;rb<padded_height;rb+=mr) {
          auto* panel=dst+((row+rb)/mr)*mr*kp;
          for(int64_t r=0;r<mr;r+=4) {
            int64_t c=0;
            if(rb+r+4<=height) {
              for(;c+8<=width;c+=8) {
                uint32x4_t v[4];
                for(int rr=0;rr<4;++rr) {
                  auto off=(rb+r+rr)*nt+c;
                  v[rr]=vreinterpretq_u32_u16(vcombine_u16(fused4(gate+off,up+off,lut),fused4(gate+off+4,up+off+4,lut)));
                }
                auto a0=vreinterpretq_u64_u32(vtrn1q_u32(v[0],v[1]));
                auto a1=vreinterpretq_u64_u32(vtrn2q_u32(v[0],v[1]));
                auto b0=vreinterpretq_u64_u32(vtrn1q_u32(v[2],v[3]));
                auto b1=vreinterpretq_u64_u32(vtrn2q_u32(v[2],v[3]));
                vst1q_u16(panel+(col+c)*mr+2*r,vreinterpretq_u16_u64(vtrn1q_u64(a0,b0)));
                vst1q_u16(panel+(col+c+2)*mr+2*r,vreinterpretq_u16_u64(vtrn1q_u64(a1,b1)));
                vst1q_u16(panel+(col+c+4)*mr+2*r,vreinterpretq_u16_u64(vtrn2q_u64(a0,b0)));
                vst1q_u16(panel+(col+c+6)*mr+2*r,vreinterpretq_u16_u64(vtrn2q_u64(a1,b1)));
              }
            }
            for(int64_t rr=r;rr<r+4;++rr)for(int64_t cc=c;cc<padded_width;++cc) {
              uint16_t value=0;
              if(rb+rr<height && cc<width) {
                auto g=at::BFloat16(gate[(rb+rr)*nt+cc]);
                auto u=at::BFloat16(up[(rb+rr)*nt+cc]);
                auto s=at::BFloat16(lut[g.x],at::BFloat16::from_bits());
                value=at::BFloat16(float(s)*float(u)).x;
              }
              panel[((col+cc)/2)*2*mr+2*rr+(col+cc)%2]=value;
            }
          }
        }
      }
    }
  });
  return out;
}
}
TORCH_LIBRARY(qwen21_w8_swiglu,m) {
  m.def("run(Tensor lhs, Tensor gate, Tensor up, Tensor table, int m, int n, int k, int workers, int mt=64, int nt=256) -> Tensor",&run);
}
