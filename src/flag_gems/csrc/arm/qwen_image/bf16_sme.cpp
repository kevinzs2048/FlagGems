#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "kernel_parallel.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <atomic>
#include <arm_neon.h>
#include <pthread.h>
#include <pthread/qos.h>
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme2.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"

#define GEMM(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa
#define RHS(name) kai_##name##_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme
#define LHS(name) kai_##name##_lhs_pack_bf16p2vlx2_f32_sme2
#define NG(name) kai_##name##_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla
#define NP(name) kai_##name##_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon
#define NL(name) kai_##name##_lhs_quant_pack_bf16p8x4_f32_neon

namespace {
int64_t ceil_div(int64_t x, int64_t y) { return (x+y-1)/y; }
// Process-local experiment. A QoS class is an OS scheduling hint, not CPU pinning.
std::tuple<std::vector<int64_t>,std::vector<int64_t>> pool_qos(int64_t requested) {
  TORCH_CHECK(requested==-1 || requested==QOS_CLASS_BACKGROUND || requested==QOS_CLASS_UTILITY ||
    requested==QOS_CLASS_DEFAULT || requested==QOS_CLASS_USER_INITIATED || requested==QOS_CLASS_USER_INTERACTIVE,
    "Unsupported QoS class");
  const int64_t threads=at::get_num_threads();
  std::vector<int64_t> before(threads,-1),after(threads,-1);
  kernel_parallel::parallel_for(0,threads,1,[&](int64_t begin,int64_t end) {
    for(int64_t i=begin;i<end;++i) {
      qos_class_t current;int relative;
      TORCH_CHECK(pthread_get_qos_class_np(pthread_self(),&current,&relative)==0,"Cannot query QoS");
      before[i]=current;
      if(requested!=-1)TORCH_CHECK(pthread_set_qos_class_self_np(static_cast<qos_class_t>(requested),0)==0,"Cannot set QoS");
      TORCH_CHECK(pthread_get_qos_class_np(pthread_self(),&current,&relative)==0,"Cannot query changed QoS");
      after[i]=current;
    }
  });
  return {before,after};
}
std::vector<int64_t> pool_qos_pattern(const std::vector<int64_t>& classes) {
  TORCH_CHECK(classes.size()==static_cast<size_t>(at::get_num_threads()),"One QoS class per Torch pool rank required");
  for(auto q:classes)TORCH_CHECK(q==9 || q==17 || q==21 || q==25 || q==33,"Invalid QoS class");
  std::vector<int64_t> actual(classes.size(),-1);
  at::parallel_for(0,classes.size(),1,[&](int64_t begin,int64_t end) {
    for(int64_t rank=begin;rank<end;++rank) {
      TORCH_CHECK(pthread_set_qos_class_self_np(static_cast<qos_class_t>(classes[rank]),0)==0,"Cannot set QoS pattern");
      qos_class_t q;int relative;
      TORCH_CHECK(pthread_get_qos_class_np(pthread_self(),&q,&relative)==0,"Cannot read QoS pattern");
      actual[rank]=q;
    }
  });
  return actual;
}
at::Tensor pack(const at::Tensor& weight) {
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type()==at::kBFloat16 && weight.dim()==2,
              "Expected CPU BF16 [N,K]");
  int64_t n=weight.size(0), k=weight.size(1);
  TORCH_CHECK(n>0 && k>0, "Empty weight");
  auto transposed=weight.t().to(at::kFloat).contiguous();
  auto bias=at::zeros({n}, transposed.options());
  auto result=at::empty({static_cast<int64_t>(RHS(get_rhs_packed_size)(n,k))},weight.options().dtype(at::kByte));
  RHS(run)(1,n,k,GEMM(get_nr)(),GEMM(get_kr)(),GEMM(get_sr)(),n*sizeof(float),
           transposed.const_data_ptr<float>(),bias.const_data_ptr<float>(),nullptr,result.data_ptr<uint8_t>(),0,nullptr);
  return result;
}

// Consume canonical BF16 [N,K] directly. The KAI ABI interleaves pairs
// along K across NR rows, preceded by NR FP32 bias values per block.
// Keep the library packer above as an independent layout oracle.
at::Tensor pack_direct(const at::Tensor& weight) {
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type()==at::kBFloat16 && weight.dim()==2,
              "Expected CPU BF16 [N,K]");
  const int64_t n=weight.size(0),k=weight.size(1);
  TORCH_CHECK(n>0 && k>0,"Empty weight");
  auto value=weight.contiguous();
  auto result=at::empty({static_cast<int64_t>(RHS(get_rhs_packed_size)(n,k))},weight.options().dtype(at::kByte));
  const int64_t nr=GEMM(get_nr)(),blocks=ceil_div(n,nr),kp=ceil_div(k,2)*2;
  const auto* src=reinterpret_cast<const uint16_t*>(value.const_data_ptr<at::BFloat16>());
  auto* dst=result.data_ptr<uint8_t>();
  const int64_t stride=nr*(sizeof(float)+kp*sizeof(uint16_t));
  kernel_parallel::parallel_for(0,blocks,1,[&](int64_t begin,int64_t end) {
    for(int64_t block=begin;block<end;++block) {
      auto* base=dst+block*stride;
      std::memset(base,0,nr*sizeof(float));
      auto* packed=reinterpret_cast<uint16_t*>(base+nr*sizeof(float));
      // Bounded tiles keep both the strided destination and source hot.
      for(int64_t kk=0;kk<kp;kk+=64) {
        for(int64_t row=0;row<nr;++row) {
          const int64_t col=block*nr+row;
          for(int64_t p=kk;p<std::min(kk+64,kp);p+=2) {
            const int64_t offset=p*nr+2*row;
            packed[offset]=col<n?src[col*k+p]:0;
            // Match KAI's odd-K padding: repeat the first element of the
            // row. The corresponding padded LHS lane is zero.
            packed[offset+1]=col<n?src[col*k+(p+1<k?p+1:0)]:0;
          }
        }
      }
    }
  });
  return result;
}

at::Tensor pack_lhs_direct(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()==2 && input.size(1)>0,
              "Expected CPU BF16 [M,K]");
  const int64_t m=input.size(0),k=input.size(1),mr=GEMM(get_mr)(),kp=ceil_div(k,2)*2;
  auto value=input.contiguous();
  auto result=at::empty({static_cast<int64_t>(LHS(get_lhs_packed_size)(m,k,mr,2,1))},input.options().dtype(at::kByte));
  const auto* src=reinterpret_cast<const uint16_t*>(value.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(result.data_ptr<uint8_t>());
  kernel_parallel::parallel_for(0,ceil_div(m,mr),1,[&](int64_t begin,int64_t end) {
    for(int64_t block=begin;block<end;++block) {
      auto* packed=dst+block*mr*kp;
      for(int64_t kk=0;kk<kp;kk+=64) {
        for(int64_t row=0;row<mr;++row) {
          const int64_t index=block*mr+row;
          for(int64_t p=kk;p<std::min(kk+64,kp);p+=2) {
            const int64_t offset=p*mr+2*row;
            packed[offset]=index<m?src[index*k+p]:0;
            packed[offset+1]=index<m && p+1<k?src[index*k+p+1]:0;
          }
        }
      }
    }
  });
  return result;
}

// Transpose four rows of BF16 pairs with NEON. Each 32-bit lane contains
// two unchanged BF16 encodings; no floating-point conversion is involved.
at::Tensor pack_lhs_neon(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()==2 && input.size(1)>0,
              "Expected CPU BF16 [M,K]");
  const int64_t m=input.size(0),k=input.size(1),mr=GEMM(get_mr)(),kp=ceil_div(k,2)*2;
  auto value=input.contiguous();
  auto result=at::empty({static_cast<int64_t>(LHS(get_lhs_packed_size)(m,k,mr,2,1))},input.options().dtype(at::kByte));
  const auto* src=reinterpret_cast<const uint16_t*>(value.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(result.data_ptr<uint8_t>());
  kernel_parallel::parallel_for(0,ceil_div(m,mr),1,[&](int64_t begin,int64_t end) {
    for(int64_t block=begin;block<end;++block) {
      auto* packed=dst+block*mr*kp;
      for(int64_t row=0;row<mr;row+=4) {
        int64_t p=0;
        if(row+4<=mr && block*mr+row+4<=m) {
          for(;p+8<=k;p+=8) {
            const uint16_t* base=src+(block*mr+row)*k+p;
            auto a=vreinterpretq_u32_u16(vld1q_u16(base));
            auto b=vreinterpretq_u32_u16(vld1q_u16(base+k));
            auto c=vreinterpretq_u32_u16(vld1q_u16(base+2*k));
            auto d=vreinterpretq_u32_u16(vld1q_u16(base+3*k));
            auto ab0=vreinterpretq_u64_u32(vtrn1q_u32(a,b));
            auto ab1=vreinterpretq_u64_u32(vtrn2q_u32(a,b));
            auto cd0=vreinterpretq_u64_u32(vtrn1q_u32(c,d));
            auto cd1=vreinterpretq_u64_u32(vtrn2q_u32(c,d));
            vst1q_u16(packed+p*mr+2*row,vreinterpretq_u16_u64(vtrn1q_u64(ab0,cd0)));
            vst1q_u16(packed+(p+2)*mr+2*row,vreinterpretq_u16_u64(vtrn1q_u64(ab1,cd1)));
            vst1q_u16(packed+(p+4)*mr+2*row,vreinterpretq_u16_u64(vtrn2q_u64(ab0,cd0)));
            vst1q_u16(packed+(p+6)*mr+2*row,vreinterpretq_u16_u64(vtrn2q_u64(ab1,cd1)));
          }
        }
        for(int64_t r=row;r<std::min(row+4,mr);++r) {
          const int64_t index=block*mr+r;
          for(int64_t q=p;q<kp;q+=2) {
            packed[q*mr+2*r]=index<m && q<k?src[index*k+q]:0;
            packed[q*mr+2*r+1]=index<m && q+1<k?src[index*k+q+1]:0;
          }
        }
      }
    }
  });
  return result;
}

at::Tensor pack_lhs_reference(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()==2 && input.size(1)>0,
              "Expected CPU BF16 [M,K]");
  const int64_t m=input.size(0),k=input.size(1),mr=GEMM(get_mr)();
  auto value=input.to(at::kFloat).contiguous();
  auto result=at::zeros({static_cast<int64_t>(LHS(get_lhs_packed_size)(m,k,mr,2,1))},input.options().dtype(at::kByte));
  if(m) LHS(run)(m,k,mr,2,1,0,value.const_data_ptr<float>(),k*sizeof(float),result.data_ptr<uint8_t>());
  return result;
}

at::Tensor linear_impl(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t k,int64_t workers,int64_t axis,bool direct,int64_t mt=0,int64_t nt=0,const std::vector<int64_t>& ranks={},bool dynamic=false,bool neon_pack=false) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()==2 && input.size(1)==k,
              "Expected CPU BF16 [M,K]");
  TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
              n>0 && k>0 && rhs.numel()==RHS(get_rhs_packed_size)(n,k),"Invalid packed weight");
  TORCH_CHECK(workers>0 && workers<=kernel_parallel::limit() && (axis==0 || axis==1),"Invalid scheduling");
  int64_t m=input.size(0);
  auto out=at::empty({m,n},input.options().dtype(at::kFloat));
  if (!m) return out;
  at::Tensor lhs;
  if(direct) lhs=neon_pack?pack_lhs_neon(input):pack_lhs_direct(input);
  else {
  auto value=input.to(at::kFloat).contiguous();
  const auto mr=GEMM(get_mr)(),kr=GEMM(get_kr)(),sr=GEMM(get_sr)();
  lhs=at::empty({static_cast<int64_t>(LHS(get_lhs_packed_size)(m,k,mr,kr,sr))},rhs.options());
  const int64_t pack_step=LHS(get_m_step)(mr),pack_tasks=std::min(workers,ceil_div(m,pack_step));
  const int64_t pack_tile=ceil_div(ceil_div(m,pack_step),pack_tasks)*pack_step;
  kernel_parallel::parallel_for(0,pack_tasks,1,[&](int64_t begin,int64_t end) {
    for(int64_t task=begin;task<end;++task) {
      int64_t row=task*pack_tile;
      if(row>=m)continue;
      LHS(run)(std::min(pack_tile,m-row),k,mr,kr,sr,0,value.const_data_ptr<float>()+row*k,k*sizeof(float),
               lhs.data_ptr<uint8_t>()+LHS(get_lhs_packed_offset)(row,k,mr,kr,sr));
    }
  });
  }
  if(mt || nt) {
    TORCH_CHECK(mt>0 && nt>0 && mt%GEMM(get_m_step)()==0 && nt%GEMM(get_n_step)()==0,"Invalid GEMM tile");
    const int64_t rows=ceil_div(m,mt),cols=ceil_div(n,nt);
    if(dynamic) {
      std::atomic<int64_t> next{0};
      kernel_parallel::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
        for(int64_t slot=begin;slot<end;++slot) {
          int64_t task;
          while((task=next.fetch_add(1,std::memory_order_relaxed))<rows*cols) {
            const int64_t row=(task%rows)*mt,col=(task/rows)*nt;
            GEMM(run)(std::min(mt,m-row),std::min(nt,n-col),k,
              lhs.const_data_ptr<uint8_t>()+GEMM(get_lhs_packed_offset)(row,k),
              rhs.const_data_ptr<uint8_t>()+GEMM(get_rhs_packed_offset)(col,k),
              out.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
          }
        }
      });
      return out;
    }
    kernel_parallel::parallel_for(0,rows*cols,1,[&](int64_t begin,int64_t end) {
      for(int64_t task=begin;task<end;++task) {
        const int64_t row=(task%rows)*mt,col=(task/rows)*nt;
        GEMM(run)(std::min(mt,m-row),std::min(nt,n-col),k,
          lhs.const_data_ptr<uint8_t>()+GEMM(get_lhs_packed_offset)(row,k),
          rhs.const_data_ptr<uint8_t>()+GEMM(get_rhs_packed_offset)(col,k),
          out.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      }
    });
    return out;
  }
  const int64_t step=axis==0?GEMM(get_m_step)():GEMM(get_n_step)();
  const int64_t extent=axis==0?m:n,tasks=std::min(workers,ceil_div(extent,step));
  const int64_t tile=ceil_div(ceil_div(extent,step),tasks)*step;
  std::vector<int64_t> task_map;
  if(!ranks.empty()) {
    TORCH_CHECK(ranks.size()==workers,"One pool rank per worker required");
    task_map.assign(at::get_num_threads(),-1);
    for(int64_t i=0;i<workers;++i) {
      TORCH_CHECK(ranks[i]>=0 && ranks[i]<at::get_num_threads() && task_map[ranks[i]]==-1,"Invalid/duplicate logical pool rank");
      task_map[ranks[i]]=i;
    }
  }
  const int64_t dispatch_size=ranks.empty()?tasks:at::get_num_threads();
  kernel_parallel::parallel_for(0,dispatch_size,1,[&](int64_t begin,int64_t end) {
    for(int64_t index=begin;index<end;++index) {
      const int64_t task=ranks.empty()?index:task_map[index];
      if(task<0 || task>=tasks)continue;
      int64_t start=task*tile;
      if(start>=extent)continue;
      int64_t row=axis==0?start:0,col=axis==1?start:0,count=std::min(tile,extent-start);
      GEMM(run)(axis==0?count:m,axis==1?count:n,k,
          lhs.const_data_ptr<uint8_t>()+GEMM(get_lhs_packed_offset)(row,k),
          rhs.const_data_ptr<uint8_t>()+GEMM(get_rhs_packed_offset)(col,k),
          out.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
    }
  });
  return out;
}
at::Tensor linear(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t k,int64_t workers,int64_t axis) {
  return linear_impl(input,rhs,n,k,workers,axis,false);
}
at::Tensor linear_direct(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t k,int64_t workers,int64_t axis) {
  return linear_impl(input,rhs,n,k,workers,axis,true);
}
at::Tensor linear_tiled(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t k,int64_t workers,int64_t mt,int64_t nt) {
  return linear_impl(input,rhs,n,k,workers,0,false,mt,nt);
}
at::Tensor linear_dynamic(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t k,int64_t workers,int64_t mt,int64_t nt) {
  TORCH_CHECK(mt>0 && nt>0,"Dynamic GEMM requires positive tile sizes");
  return linear_impl(input,rhs,n,k,workers,0,false,mt,nt,{},true);
}
at::Tensor linear_dynamic_neon_pack(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t k,int64_t workers,int64_t mt,int64_t nt) {
  TORCH_CHECK(mt>0 && nt>0,"Dynamic GEMM requires positive tile sizes");
  return linear_impl(input,rhs,n,k,workers,0,true,mt,nt,{},true,true);
}
at::Tensor linear_pool_ranks(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t k,std::vector<int64_t> ranks,int64_t axis) {
  TORCH_CHECK(!ranks.empty(),"Empty rank list");
  return linear_impl(input,rhs,n,k,ranks.size(),axis,false,0,0,ranks);
}
at::Tensor pack_neon(const at::Tensor& weight) {
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type()==at::kBFloat16 && weight.dim()==2 &&
              weight.size(0)>0 && weight.size(1)>0,"Expected nonempty CPU BF16 weight");
  const int64_t n=weight.size(0),k=weight.size(1),nr=NG(get_nr)(),kr=NG(get_kr)();
  auto x=weight.t().to(at::kFloat).contiguous(),bias=at::zeros({n},x.options());
  auto out=at::empty({static_cast<int64_t>(NP(get_rhs_packed_size)(n,k,nr,kr))},weight.options().dtype(at::kByte));
  kernel_parallel::parallel_for(0,ceil_div(n,nr),1,[&](int64_t begin,int64_t end) {
    const int64_t col=begin*nr,count=std::min(end*nr,n)-col;
    NP(run)(1,count,k,nr,kr,1,n*sizeof(float),x.const_data_ptr<float>()+col,bias.const_data_ptr<float>()+col,
      nullptr,out.data_ptr<uint8_t>()+NP(get_rhs_packed_offset)(col,k,nr,kr),0,nullptr);
  });
  return out;
}
// BF16 values preserved in both layouts. Accumulation order can differ between
// BFMOPA and BFMMLA; validate against floating references, not quantized ones.
at::Tensor linear_hybrid(const at::Tensor& input,const at::Tensor& ps,const at::Tensor& pn,
                        int64_t ns,int64_t nn,int64_t sw,int64_t nw) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type()==at::kBFloat16 && input.dim()==2 && input.size(1)>0,
              "Expected CPU BF16 activation");
  const int64_t m=input.size(0),k=input.size(1),n=ns+nn;
  TORCH_CHECK(ns>0 && nn>0 && ns%GEMM(get_n_step)()==0 && sw>0 && nw>0 && sw+nw<=kernel_parallel::limit(),"Invalid hybrid BF16 split/workers");
  TORCH_CHECK(ps.device().is_cpu() && ps.scalar_type()==at::kByte && ps.dim()==1 && ps.is_contiguous() &&
    ps.numel()==RHS(get_rhs_packed_size)(ns,k),"Invalid SME BF16 packed weight");
  TORCH_CHECK(pn.device().is_cpu() && pn.scalar_type()==at::kByte && pn.dim()==1 && pn.is_contiguous() &&
    pn.numel()==NP(get_rhs_packed_size)(nn,k,NG(get_nr)(),NG(get_kr)()),"Invalid NEON BF16 packed weight");
  auto out=at::empty({m,n},input.options().dtype(at::kFloat));if(!m)return out;
  auto x=input.to(at::kFloat).contiguous();
  const int64_t sm=GEMM(get_mr)(),nm=NG(get_mr)();
  auto ls=at::zeros({static_cast<int64_t>(LHS(get_lhs_packed_size)(m,k,sm,2,1))},ps.options());
  auto ln=at::zeros({static_cast<int64_t>(NL(get_lhs_packed_size)(m,k,nm,4,1))},pn.options());
  kernel_parallel::parallel_for(0,ceil_div(m,sm),1,[&](int64_t b,int64_t e) {
    int64_t row=b*sm,count=std::min(e*sm,m)-row;
    LHS(run)(count,k,sm,2,1,0,x.const_data_ptr<float>()+row*k,k*sizeof(float),ls.data_ptr<uint8_t>()+LHS(get_lhs_packed_offset)(row,k,sm,2,1));
  });
  kernel_parallel::parallel_for(0,ceil_div(m,nm),1,[&](int64_t b,int64_t e) {
    int64_t row=b*nm,count=std::min(e*nm,m)-row;
    NL(run)(count,k,nm,4,1,0,x.const_data_ptr<float>()+row*k,k*sizeof(float),ln.data_ptr<uint8_t>()+NL(get_lhs_packed_offset)(row,k,nm,4,1));
  });
  const int64_t st=ceil_div(ceil_div(ns,GEMM(get_n_step)()),sw)*GEMM(get_n_step)();
  const int64_t nt=ceil_div(ceil_div(nn,NG(get_n_step)()),nw)*NG(get_n_step)();
  kernel_parallel::parallel_for(0,sw+nw,1,[&](int64_t b,int64_t e) {
    for(int64_t task=b;task<e;++task) {
      if(task<sw) {
        int64_t col=task*st;if(col>=ns)continue;
        GEMM(run)(m,std::min(st,ns-col),k,ls.const_data_ptr<uint8_t>(),ps.const_data_ptr<uint8_t>()+GEMM(get_rhs_packed_offset)(col,k),
          out.data_ptr<float>()+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      } else {
        int64_t col=(task-sw)*nt;if(col>=nn)continue;
        NG(run)(m,std::min(nt,nn-col),k,ln.const_data_ptr<uint8_t>(),pn.const_data_ptr<uint8_t>()+NG(get_rhs_packed_offset)(col,k),
          out.data_ptr<float>()+ns+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      }
    }
  });
  return out;
}
}
TORCH_LIBRARY(qwen21_bf16_sme,m) {
  m.def("pool_qos_pattern(int[] classes) -> int[]",&pool_qos_pattern);
  m.def("configure_parallel(bool kernel_owned, int workers) -> ()",&kernel_parallel::configure);
  m.def("parallel_info() -> int[]",&kernel_parallel::info);
  m.def("pool_qos(int requested) -> (int[], int[])",&pool_qos);
  m.def("pack(Tensor weight) -> Tensor",&pack);
  m.def("pack_direct(Tensor weight) -> Tensor",&pack_direct);
  m.def("pack_lhs_reference(Tensor input) -> Tensor",&pack_lhs_reference);
  m.def("pack_lhs_direct(Tensor input) -> Tensor",&pack_lhs_direct);
  m.def("pack_lhs_neon(Tensor input) -> Tensor",&pack_lhs_neon);
  m.def("linear(Tensor input, Tensor packed, int n, int k, int workers, int axis) -> Tensor",&linear);
  m.def("linear_direct(Tensor input, Tensor packed, int n, int k, int workers, int axis) -> Tensor",&linear_direct);
  m.def("linear_tiled(Tensor input, Tensor packed, int n, int k, int workers, int mt, int nt) -> Tensor",&linear_tiled);
  m.def("linear_dynamic(Tensor input, Tensor packed, int n, int k, int workers, int mt, int nt) -> Tensor",&linear_dynamic);
  m.def("linear_dynamic_neon_pack(Tensor input, Tensor packed, int n, int k, int workers, int mt, int nt) -> Tensor",&linear_dynamic_neon_pack);
  m.def("linear_pool_ranks(Tensor input, Tensor packed, int n, int k, int[] ranks, int axis) -> Tensor",&linear_pool_ranks);
  m.def("pack_neon(Tensor weight) -> Tensor",&pack_neon);
  m.def("linear_hybrid(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int sme_workers, int neon_workers) -> Tensor",&linear_hybrid);
}
