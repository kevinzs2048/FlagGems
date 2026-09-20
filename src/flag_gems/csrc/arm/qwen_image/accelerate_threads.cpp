// Accelerate's BLAS policy is thread-local; configure every OpenMP worker.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <Accelerate/Accelerate.h>
#include <omp.h>
#include <pthread.h>

namespace {
at::Tensor visit(int64_t mode,const at::Tensor* restore=nullptr) {
  TORCH_CHECK(mode>=-1 && mode<=1,"BLAS policy must be query(-1), multi(0), or single(1)");
  TORCH_CHECK(!omp_in_parallel(),"Configure BLAS policy outside parallel work");
  if(__builtin_available(macOS 15.0,*)) {
    const int64_t count=at::get_num_threads();int64_t active=0;
    const int64_t* previous=nullptr;int64_t old_count=0;
    if(restore) {
      TORCH_CHECK(restore->device().is_cpu() && restore->scalar_type()==at::kLong && restore->dim()==2 &&
                  restore->size(1)==5 && restore->is_contiguous(),"Expected saved BLAS worker state");
      previous=restore->const_data_ptr<int64_t>();old_count=restore->size(0);
    }
    auto result=at::empty({count,5},at::TensorOptions().dtype(at::kLong));
    auto* rows=result.data_ptr<int64_t>();
    #pragma omp parallel num_threads(count)
    {
      const int64_t rank=omp_get_thread_num();
      #pragma omp single
      {active=omp_get_num_threads();}
      uint64_t tid=0;const int id_status=pthread_threadid_np(nullptr,&tid);
      const int64_t before=BLASGetThreading();int status=0;int64_t requested=mode;
      if(restore) {
        requested=-1;status=-2;
        for(int64_t i=0;i<old_count;++i) {
          if(id_status==0 && previous[i*5]==static_cast<int64_t>(tid)) {
            requested=previous[i*5+2];status=0;break;
          }
        }
      }
      if(id_status!=0)status=-3;
      if(status==0 && requested>=0)status=BLASSetThreading(static_cast<BLAS_THREADING>(requested));
      auto* row=rows+rank*5;row[0]=static_cast<int64_t>(tid);row[1]=rank;row[2]=before;
      row[3]=status;row[4]=BLASGetThreading();
    }
    return result.slice(0,0,active);
  }
  TORCH_CHECK(false,"Accelerate BLAS thread policy requires macOS 15 or later");
}
at::Tensor pool_mode(int64_t mode) {return visit(mode);}
at::Tensor restore_pool(const at::Tensor& state) {return visit(-1,&state);}
}
TORCH_LIBRARY(qwen21_accelerate_threads,m) {
  m.def("pool_mode(int mode) -> Tensor",&pool_mode);
  m.def("restore_pool(Tensor state) -> Tensor",&restore_pool);
}
