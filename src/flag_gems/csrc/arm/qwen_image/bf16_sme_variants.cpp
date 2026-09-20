// Experimental packed GEMM comparison. Original KAI packing and complete-K
// accumulation are shared across variants; only microkernel and task order vary.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <atomic>
#include <cfloat>
#include <pthread.h>
#include <mach/mach_time.h>
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa.h"
#define V2(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa
#define V1(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa

namespace {
int64_t divup(int64_t x,int64_t y) {return (x+y-1)/y;}
// Arithmetic-only diagnostic, NOT a GEMM implementation. All multiplicands
// stay in streaming registers. Four independent ZA tiles follow the same
// dependency pattern as the production 2vl x 2vl BF16 kernel.
int64_t arithmetic_probe(int64_t workers,int64_t iterations,bool int8) {
  TORCH_CHECK(workers>0 && workers<=at::get_num_threads() && iterations>0 && iterations<=100000000,
              "Invalid arithmetic probe size");
  const int64_t vl=V2(get_mr)()/2;
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t task=begin;task<end;++task) {
      int64_t count=iterations;
      const int value=int8?0x0101:0x3f80; // INT8 ones / BF16 1.0.
      asm volatile(
        ".arch armv9-a+sme2\n"
        // Enable both streaming mode and ZA. `smstart za` alone leaves SM=0;
        // Apple exposes streaming SVE, not ordinary non-streaming SVE.
        ".inst 0xd503477f\n"
        "ptrue p0.b\n"
        "dup z4.h, %w[value]\n"
        "dup z20.h, %w[value]\n"
        "zero {za}\n"
        "cbz %w[int8], 2f\n"
        "1:\n"
        ".rept 4\n"
        "smopa za0.s, p0/m, p0/m, z4.b, z20.b\n"
        "smopa za1.s, p0/m, p0/m, z4.b, z20.b\n"
        "smopa za2.s, p0/m, p0/m, z4.b, z20.b\n"
        "smopa za3.s, p0/m, p0/m, z4.b, z20.b\n"
        ".endr\n"
        "subs %x[count], %x[count], #1\n"
        "b.ne 1b\n"
        "b 3f\n"
        "2:\n"
        ".rept 4\n"
        "bfmopa za0.s, p0/m, p0/m, z4.h, z20.h\n"
        "bfmopa za1.s, p0/m, p0/m, z4.h, z20.h\n"
        "bfmopa za2.s, p0/m, p0/m, z4.h, z20.h\n"
        "bfmopa za3.s, p0/m, p0/m, z4.h, z20.h\n"
        ".endr\n"
        "subs %x[count], %x[count], #1\n"
        "b.ne 2b\n"
        "3:\n"
        "smstop\n"
        :[count] "+r"(count):[value] "r"(value),[int8] "r"(static_cast<int>(int8))
        :"cc","memory","v0","v1","v2","v3","v4","v5","v6","v7",
         "v8","v9","v10","v11","v12","v13","v14","v15",
         "v16","v17","v18","v19","v20","v21","v22","v23",
         "v24","v25","v26","v27","v28","v29","v30","v31");
    }
  });
  // Each BFMOPA performs two products and two adds per FP32 output lane.
  return workers*iterations*16*(int8?8:4)*vl*vl;
}
at::Tensor matmul(const at::Tensor& lhs,const at::Tensor& rhs,int64_t m,int64_t n,int64_t k,
                  int64_t workers,int64_t mt,int64_t nt,int64_t version,int64_t order) {
  TORCH_CHECK(m>=0 && n>0 && k>0 && workers>0 && workers<=at::get_num_threads(),"Invalid dimensions or workers");
  TORCH_CHECK((version==1 || version==2) && order>=0 && order<=3,"Invalid kernel/order");
  const int64_t mr=V2(get_mr)(),nr=V2(get_nr)();
  TORCH_CHECK(mr==V1(get_mr)() && nr==V1(get_nr)(),"Incompatible kernel layouts");
  TORCH_CHECK(mt>0 && nt>0 && mt%mr==0 && nt%nr==0,"Invalid output tile");
  TORCH_CHECK(lhs.device().is_cpu() && lhs.scalar_type()==at::kByte && lhs.dim()==1 && lhs.is_contiguous() &&
              lhs.numel()==divup(m,mr)*mr*divup(k,2)*2*2,"Invalid packed LHS");
  TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
              rhs.numel()==divup(n,nr)*nr*(4+divup(k,2)*2*2),"Invalid packed RHS");
  auto out=at::empty({m,n},lhs.options().dtype(at::kFloat));
  if(!m)return out;
  auto kernel=version==1?V1(run):V2(run);
  const int64_t rows=divup(m,mt),cols=divup(n,nt),group=order==2?2:4;
  // Orders 2/3 cover a padded grid of 2x2/4x4 output tiles, skipping edges.
  const int64_t total=order<2?rows*cols:divup(rows,group)*divup(cols,group)*group*group;
  std::atomic<int64_t> next{0};
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      int64_t task;
      while((task=next.fetch_add(1,std::memory_order_relaxed))<total) {
        int64_t r,c;
        if(order==0) {r=task%rows;c=task/rows;}
        else if(order==1) {r=task/cols;c=task%cols;}
        else {
          const int64_t tile=task/(group*group),inside=task%(group*group);
          r=(tile%divup(rows,group))*group+inside%group;
          c=(tile/divup(rows,group))*group+inside/group;
        }
        if(r>=rows || c>=cols)continue;
        r*=mt;c*=nt;
        kernel(std::min(mt,m-r),std::min(nt,n-c),k,
               lhs.const_data_ptr<uint8_t>()+V2(get_lhs_packed_offset)(r,k),
               rhs.const_data_ptr<uint8_t>()+V2(get_rhs_packed_offset)(c,k),
               out.data_ptr<float>()+r*n+c,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      }
    }
  });
  return out;
}
// Diagnostic copy: production matmul above contains no tracing branches.
std::tuple<at::Tensor,at::Tensor> matmul_trace(const at::Tensor& lhs,const at::Tensor& rhs,int64_t m,int64_t n,int64_t k,
                  int64_t workers,int64_t mt,int64_t nt,int64_t version,int64_t order) {
  TORCH_CHECK(m>=0 && n>0 && k>0 && workers>0 && workers<=at::get_num_threads(),"Invalid dimensions or workers");
  TORCH_CHECK((version==1 || version==2) && order>=0 && order<=3,"Invalid kernel/order");
  const int64_t mr=V2(get_mr)(),nr=V2(get_nr)();
  TORCH_CHECK(mr==V1(get_mr)() && nr==V1(get_nr)(),"Incompatible kernel layouts");
  TORCH_CHECK(mt>0 && nt>0 && mt%mr==0 && nt%nr==0,"Invalid output tile");
  TORCH_CHECK(lhs.device().is_cpu() && lhs.scalar_type()==at::kByte && lhs.dim()==1 && lhs.is_contiguous() &&
              lhs.numel()==divup(m,mr)*mr*divup(k,2)*2*2,"Invalid packed LHS");
  TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
              rhs.numel()==divup(n,nr)*nr*(4+divup(k,2)*2*2),"Invalid packed RHS");
  auto out=at::empty({m,n},lhs.options().dtype(at::kFloat));
  const int64_t trace_rows=divup(m,mt),trace_cols=divup(n,nt);
  auto trace=at::empty({trace_rows*trace_cols,8},lhs.options().dtype(at::kLong));
  if(!m)return {out,trace};
  auto* records=trace.data_ptr<int64_t>();
  auto cpu_number=[]() -> int64_t {size_t cpu=0;return pthread_cpu_number_np(&cpu)==0?static_cast<int64_t>(cpu):-1;};
  auto kernel=version==1?V1(run):V2(run);
  const int64_t rows=divup(m,mt),cols=divup(n,nt),group=order==2?2:4;
  // Orders 2/3 cover a padded grid of 2x2/4x4 output tiles, skipping edges.
  const int64_t total=order<2?rows*cols:divup(rows,group)*divup(cols,group)*group*group;
  std::atomic<int64_t> next{0};
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      int64_t task;
      while((task=next.fetch_add(1,std::memory_order_relaxed))<total) {
        int64_t r,c;
        if(order==0) {r=task%rows;c=task/rows;}
        else if(order==1) {r=task/cols;c=task%cols;}
        else {
          const int64_t tile=task/(group*group),inside=task%(group*group);
          r=(tile%divup(rows,group))*group+inside%group;
          c=(tile/divup(rows,group))*group+inside/group;
        }
        if(r>=rows || c>=cols)continue;
        const int64_t index=r*cols+c;
        r*=mt;c*=nt;
        const int64_t before=cpu_number();const uint64_t started=mach_absolute_time();
        kernel(std::min(mt,m-r),std::min(nt,n-c),k,
               lhs.const_data_ptr<uint8_t>()+V2(get_lhs_packed_offset)(r,k),
               rhs.const_data_ptr<uint8_t>()+V2(get_rhs_packed_offset)(c,k),
               out.data_ptr<float>()+r*n+c,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
        const uint64_t finished=mach_absolute_time();const int64_t after=cpu_number();
        auto* p=records+index*8;p[0]=0;p[1]=slot;p[2]=r;p[3]=c;p[4]=before;p[5]=after;
        p[6]=static_cast<int64_t>(started);p[7]=static_cast<int64_t>(finished);
      }
    }
  });
  return {out,trace};
}
}
TORCH_LIBRARY(qwen21_bf16_variants,m) {
  m.def("packed_rows() -> int",[](){return static_cast<int64_t>(V2(get_mr)());});
  m.def("arithmetic_probe(int workers, int iterations, bool int8=False) -> int",&arithmetic_probe);
  m.def("matmul_trace(Tensor lhs, Tensor rhs, int m, int n, int k, int workers, int mt, int nt, int version, int order) -> (Tensor, Tensor)",&matmul_trace);
  m.def("matmul(Tensor lhs, Tensor rhs, int m, int n, int k, int workers, int mt, int nt, int version, int order) -> Tensor",&matmul);
}
