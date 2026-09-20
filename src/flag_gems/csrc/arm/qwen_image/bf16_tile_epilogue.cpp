// Experimental only: retain the existing full-K KAI arithmetic, but round each
// completed output tile into BF16 before leaving its worker's scratch buffer.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <atomic>
#include <cfloat>
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#define GEMM(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa

namespace {
int64_t divup(int64_t x,int64_t y) {return (x+y-1)/y;}
uint16x4_t rounded(float32x4_t value) {
  auto bits=vreinterpretq_u32_f32(value);
  auto r=vshrq_n_u32(vaddq_u32(bits,vaddq_u32(vdupq_n_u32(0x7fff),
                    vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1)))),16);
  return vmovn_u32(vbslq_u32(vmvnq_u32(vceqq_f32(value,value)),vdupq_n_u32(0x7fc0),r));
}
at::Tensor matmul_bf16(const at::Tensor& lhs,const at::Tensor& rhs,int64_t m,int64_t n,int64_t k,
                       int64_t workers,int64_t mt,int64_t nt) {
  TORCH_CHECK(m>=0 && n>0 && k>0 && workers>0 && workers<=at::get_num_threads(),"Invalid dimensions/workers");
  const int64_t mr=GEMM(get_mr)(),nr=GEMM(get_nr)();
  TORCH_CHECK(mt>0 && mt<=256 && nt>0 && nt<=2048 && mt%mr==0 && nt%nr==0,"Invalid tile size");
  TORCH_CHECK(lhs.device().is_cpu() && lhs.scalar_type()==at::kByte && lhs.dim()==1 && lhs.is_contiguous() &&
              lhs.numel()==divup(m,mr)*mr*divup(k,2)*2*2,"Invalid packed LHS");
  TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
              rhs.numel()==divup(n,nr)*nr*(4+divup(k,2)*2*2),"Invalid packed RHS");
  auto out=at::empty({m,n},lhs.options().dtype(at::kBFloat16));
  if(!m)return out;
  auto scratch=at::empty({workers,mt*nt},lhs.options().dtype(at::kFloat));
  const int64_t rows=divup(m,mt),cols=divup(n,nt);
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  std::atomic<int64_t> next{0};
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      auto* tile=scratch.data_ptr<float>()+slot*mt*nt;
      int64_t task;
      while((task=next.fetch_add(1,std::memory_order_relaxed))<rows*cols) {
        const int64_t row=task%rows*mt,col=task/rows*nt;
        const int64_t height=std::min(mt,m-row),width=std::min(nt,n-col);
        GEMM(run)(height,width,k,
          lhs.const_data_ptr<uint8_t>()+GEMM(get_lhs_packed_offset)(row,k),
          rhs.const_data_ptr<uint8_t>()+GEMM(get_rhs_packed_offset)(col,k),
          tile,nt*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
        for(int64_t r=0;r<height;++r) {
          int64_t c=0;
          for(;c+4<=width;c+=4)vst1_u16(dst+(row+r)*n+col+c,rounded(vld1q_f32(tile+r*nt+c)));
          for(;c<width;++c)dst[(row+r)*n+col+c]=at::BFloat16(tile[r*nt+c]).x;
        }
      }
    }
  });
  return out;
}
}
TORCH_LIBRARY(qwen21_bf16_epilogue,m) {
  m.def("matmul(Tensor lhs, Tensor rhs, int m, int n, int k, int workers, int mt, int nt) -> Tensor",&matmul_bf16);
}
