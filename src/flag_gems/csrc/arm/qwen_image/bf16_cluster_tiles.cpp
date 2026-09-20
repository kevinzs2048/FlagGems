// Experimental full-K GEMM scheduling: smaller claims on slower CPU clusters.
// CPU number is an instantaneous sample, not a pinning or affinity guarantee.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <atomic>
#include <cfloat>
#include <pthread.h>
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#define G(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa

namespace {
int64_t divup(int64_t x,int64_t y) {return (x+y-1)/y;}
at::Tensor matmul(const at::Tensor& lhs,const at::Tensor& rhs,int64_t m,int64_t n,int64_t k,
                  int64_t workers,int64_t mt,int64_t nt,const at::Tensor& cpu_clusters,
                  const at::Tensor& cluster_batches,int64_t order) {
  TORCH_CHECK(m>=0 && n>0 && k>0 && workers>0 && workers<=at::get_num_threads(),"Invalid dimensions/workers");
  TORCH_CHECK(order==0 || order==1,"Order must be M-major or N-major");
  const int64_t mr=G(get_mr)(),nr=G(get_nr)();
  TORCH_CHECK(mt>0 && nt>0 && mt%mr==0 && nt%nr==0,"Invalid tile");
  TORCH_CHECK(lhs.device().is_cpu() && lhs.scalar_type()==at::kByte && lhs.dim()==1 && lhs.is_contiguous() &&
              lhs.numel()==divup(m,mr)*mr*divup(k,2)*2*2,"Invalid packed LHS");
  TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
              rhs.numel()==divup(n,nr)*nr*(4+divup(k,2)*2*2),"Invalid packed RHS");
  for(const auto& t:{cpu_clusters,cluster_batches})
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==at::kLong && t.dim()==1 && t.is_contiguous() && t.numel()>0,
                "Expected nonempty CPU int64 topology/policy");
  const auto* clusters=cpu_clusters.const_data_ptr<int64_t>();const auto* batches=cluster_batches.const_data_ptr<int64_t>();
  const int64_t ncpu=cpu_clusters.numel(),ncluster=cluster_batches.numel();
  bool uniform=true;
  for(int64_t i=0;i<ncpu;++i)TORCH_CHECK(clusters[i]>=0 && clusters[i]<ncluster,"Invalid cluster id");
  for(int64_t i=0;i<ncluster;++i) {
    TORCH_CHECK(batches[i]>=1 && batches[i]<=16,"Claim size must be 1..16");
    uniform=uniform && batches[i]==batches[0];
  }
  auto output=at::empty({m,n},lhs.options().dtype(at::kFloat));
  if(!m)return output;
  const int64_t rows=divup(m,mt),cols=divup(n,nt),total=rows*cols;
  std::atomic<int64_t> next{0};
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      while(true) {
        int64_t claim=batches[0];
        if(!uniform) {
          size_t cpu=0;
          claim=pthread_cpu_number_np(&cpu)==0 && cpu<static_cast<size_t>(ncpu)?batches[clusters[cpu]]:1;
        }
        const int64_t first=next.fetch_add(claim,std::memory_order_relaxed);
        if(first>=total)break;
        const int64_t last=std::min(first+claim,total);
        for(int64_t task=first;task<last;) {
          const int64_t row=(order==0?task%rows:task/cols)*mt;
          const int64_t col=(order==0?task/rows:task%cols)*nt;
          const int64_t count=std::min(last-task,order==0?rows-task%rows:cols-task%cols);
          const int64_t mm=std::min((order==0?count:1)*mt,m-row);
          const int64_t nn=std::min((order==1?count:1)*nt,n-col);
          G(run)(mm,nn,k,lhs.const_data_ptr<uint8_t>()+G(get_lhs_packed_offset)(row,k),
                 rhs.const_data_ptr<uint8_t>()+G(get_rhs_packed_offset)(col,k),
                 output.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
          task+=count;
        }
      }
    }
  });
  return output;
}
}
TORCH_LIBRARY(qwen21_bf16_cluster,m) {
  m.def("matmul(Tensor lhs, Tensor rhs, int m, int n, int k, int workers, int mt, int nt, Tensor cpu_clusters, Tensor cluster_batches, int order) -> Tensor",&matmul);
}
