// Experimental direct BF16 stores from streaming vector registers.
// Full-K arithmetic and input ABI match the KAI BF16 reference.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "kai/kai_common.h"
#include <atomic>
#include <cfloat>
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#define GEMM(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa

struct KernelArgs {
 const void* A; const void* B; void* C;
 uint64_t ldcb,M,N,K;
 float min,max;
 void* accumulator_buffer;
 uint64_t flags;
};
extern "C" void qwen21_bf16_register_epilogue_kernel(KernelArgs*);
// kai_common.h hides this declaration in non-SVE translation units. Keep this
// wrapper compiled for baseline Arm; only the separate assembly uses SVE/SME.
extern "C" void kai_commit_za(void);
namespace {
int64_t divup(int64_t x,int64_t y) {return (x+y-1)/y;}
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
  const int64_t rows=divup(m,mt),cols=divup(n,nt);
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  std::atomic<int64_t> next{0};
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      int64_t task;
      while((task=next.fetch_add(1,std::memory_order_relaxed))<rows*cols) {
        const int64_t row=task%rows*mt,col=task/rows*nt;
        const int64_t height=std::min(mt,m-row),width=std::min(nt,n-col);
        KernelArgs args{
          lhs.const_data_ptr<uint8_t>()+GEMM(get_lhs_packed_offset)(row,k),
          rhs.const_data_ptr<uint8_t>()+GEMM(get_rhs_packed_offset)(col,k),
          dst+row*n+col,uint64_t(n*2),uint64_t(height),uint64_t(width),uint64_t(k),
          -FLT_MAX,FLT_MAX,nullptr,0};
        kai_commit_za();
        qwen21_bf16_register_epilogue_kernel(&args);
      }
    }
  });
  return out;
}
}
TORCH_LIBRARY(qwen21_bf16_register_epilogue,m) {
  m.def("matmul(Tensor lhs, Tensor rhs, int m, int n, int k, int workers, int mt, int nt) -> Tensor",&matmul_bf16);
}
