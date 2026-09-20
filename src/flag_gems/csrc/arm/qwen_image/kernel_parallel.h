#pragma once
#include <ATen/Parallel.h>
#include <omp.h>
#include <atomic>
#include <exception>
#include <mutex>
#include <algorithm>

// Configure once on the inference caller before requests. Each extension owns
// its configuration, but both reuse the one libomp runtime already loaded by Torch.
// No nested team and no second OpenMP library are introduced.
namespace {
namespace kernel_parallel {
static std::atomic<bool> owned{false};
static std::atomic<int64_t> team_size{18};
void configure(bool enabled,int64_t workers) {
  TORCH_CHECK(workers>0 && workers<=256,"Kernel team must have 1..256 workers");
  team_size.store(workers,std::memory_order_relaxed);
  owned.store(enabled,std::memory_order_relaxed);
}
int64_t limit() {return owned.load(std::memory_order_relaxed)?team_size.load(std::memory_order_relaxed):at::get_num_threads();}
std::vector<int64_t> info() {
  return {owned.load(),limit(),omp_get_max_threads(),omp_get_dynamic(),omp_get_num_places(),omp_get_place_num()};
}
template<class F> void parallel_for(int64_t begin,int64_t end,int64_t grain,const F& fn) {
  if(!owned.load(std::memory_order_relaxed)) {at::parallel_for(begin,end,grain,fn);return;}
  if(end<=begin)return;
  const int64_t count=std::min(limit(),(end-begin+std::max<int64_t>(1,grain)-1)/std::max<int64_t>(1,grain));
  if(count<=1 || omp_in_parallel()) {fn(begin,end);return;}
  std::exception_ptr error;std::mutex error_mutex;
  #pragma omp parallel num_threads(count)
  {
    const int64_t threads=omp_get_num_threads(),rank=omp_get_thread_num();
    const int64_t chunk=(end-begin+threads-1)/threads;
    const int64_t first=begin+rank*chunk,last=std::min(end,first+chunk);
    if(first<last) {
      try {fn(first,last);} catch(...) {
        std::lock_guard<std::mutex> guard(error_mutex);
        if(!error)error=std::current_exception();
      }
    }
  }
  if(error)std::rethrow_exception(error);
}
}
}
