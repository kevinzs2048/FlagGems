#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "kernel_parallel.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <vector>
#include <atomic>
#include <arm_neon.h>
#include <cmath>
#include <pthread.h>
#include <mach/mach_time.h>
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"

#define WG(name) kai_##name##_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa
#define WP(name) kai_##name##_rhs_pack_nxk_qsi8cxp_qsi8cx_neon
#define AP(name) kai_##name##_lhs_quant_pack_qai8dxp_f32
#define NG(name) kai_##name##_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm

namespace {
int64_t ceil_div(int64_t x,int64_t y) {return (x+y-1)/y;}
void check_k(int64_t k) {
  TORCH_CHECK(k>0 && k%8==0 && k<=32768,"W8 SME requires K%8=0, 0<K<=32768 (INT32 headroom)");
}
std::vector<int64_t> layout() {
  return {static_cast<int64_t>(WG(get_mr)()),static_cast<int64_t>(WG(get_nr)()),
          static_cast<int64_t>(WG(get_kr)()),static_cast<int64_t>(WG(get_sr)())};
}
at::Tensor pack_impl(const at::Tensor& codes,const at::Tensor& scales,bool neon) {
  TORCH_CHECK(codes.device().is_cpu() && codes.scalar_type()==at::kChar && codes.dim()==2,"Expected CPU INT8 [N,K]");
  const int64_t n=codes.size(0),k=codes.size(1);
  check_k(k);
  TORCH_CHECK(n>0 && scales.device().is_cpu() && scales.scalar_type()==at::kFloat && scales.dim()==1 && scales.numel()==n,
              "Expected CPU FP32 scales [N]");
  auto w=codes.contiguous(),s=scales.contiguous();
  const int64_t nr=neon?NG(get_nr)():WG(get_nr)(),kr=neon?NG(get_kr)():WG(get_kr)(),sr=1;
  auto result=at::empty({static_cast<int64_t>(WP(get_rhs_packed_size)(n,k,nr,kr,sr))},codes.options().dtype(at::kByte));
  const kai_rhs_pack_qsi8cx_params params={1,1.0f};
  kernel_parallel::parallel_for(0,ceil_div(n,nr),1,[&](int64_t begin,int64_t end) {
    const int64_t row=begin*nr,count=std::min(end*nr,n)-row;
    WP(run)(1,count,k,nr,kr,sr,w.const_data_ptr<int8_t>()+row*k,nullptr,s.const_data_ptr<float>()+row,
      result.data_ptr<uint8_t>()+WP(get_rhs_packed_offset)(row,k,nr,kr,sr),0,&params);
  });
  return result;
}
at::Tensor pack(const at::Tensor& codes,const at::Tensor& scales) {return pack_impl(codes,scales,false);}
at::Tensor pack_neon(const at::Tensor& codes,const at::Tensor& scales) {return pack_impl(codes,scales,true);}
at::Tensor quant_pack_impl(const at::Tensor& input,bool neon) {
  TORCH_CHECK(input.device().is_cpu() && input.dim()==2 && (input.scalar_type()==at::kBFloat16 || input.scalar_type()==at::kFloat),
              "Expected CPU BF16/FP32 [M,K]");
  const int64_t m=input.size(0),k=input.size(1);
  check_k(k);
  auto x=input.to(at::kFloat).contiguous();
  const int64_t mr=neon?NG(get_mr)():WG(get_mr)(),kr=neon?NG(get_kr)():WG(get_kr)(),sr=1;
  auto result=at::zeros({static_cast<int64_t>(AP(get_lhs_packed_size)(m,k,mr,kr,sr))},input.options().dtype(at::kByte));
  kernel_parallel::parallel_for(0,ceil_div(m,mr),1,[&](int64_t begin,int64_t end) {
    const int64_t row=begin*mr,count=std::min(end*mr,m)-row;
    AP(run)(count,k,mr,kr,sr,0,x.const_data_ptr<float>()+row*k,k*sizeof(float),
      result.data_ptr<uint8_t>()+AP(get_lhs_packed_offset)(row,k,mr,kr,sr));
  });
  return result;
}
at::Tensor quant_pack(const at::Tensor& input) {return quant_pack_impl(input,false);}
// BF16-load adaptation of KleidiAI's kai_lhs_quant_pack_qai8dxp_f32,
// Copyright 2024-2026 Arm Limited and/or its affiliates, Apache-2.0.
// Keep its min/max order, two distinct rounding rules and padding conventions.
// Only expand BF16 in vector registers; never allocate a whole FP32 activation.
at::Tensor quant_pack_bf16(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.dim()==2 && input.scalar_type()==at::kBFloat16,"Expected CPU BF16 activation");
  const int64_t m=input.size(0),k=input.size(1);check_k(k);
  auto x=input.contiguous();const auto* src=reinterpret_cast<const uint16_t*>(x.const_data_ptr<at::BFloat16>());
  const int64_t mr=WG(get_mr)(),kp=ceil_div(k,32)*32;
  TORCH_CHECK(WG(get_kr)()==4 && WG(get_sr)()==1,"Unsupported direct A8 layout");
  auto result=at::zeros({static_cast<int64_t>(AP(get_lhs_packed_size)(m,k,mr,4,1))},input.options().dtype(at::kByte));
  auto* output=result.data_ptr<uint8_t>();
  auto load4=[](const uint16_t* p) {return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p),16));};
  kernel_parallel::parallel_for(0,ceil_div(m,mr),1,[&](int64_t begin,int64_t end) {
    for(int64_t block=begin;block<end;++block) {
      auto* panel=output+block*mr*(kp+8);
      for(int64_t r=0;r<mr && block*mr+r<m;++r) {
        const auto* row=src+(block*mr+r)*k;
        auto hi=vdupq_n_f32(-FLT_MAX),lo=vdupq_n_f32(FLT_MAX);
        for(int64_t kk=0;kk<k;kk+=8) {
          auto a=load4(row+kk),b=load4(row+kk+4);
          hi=vmaxq_f32(a,hi);hi=vmaxq_f32(hi,b);
          lo=vminq_f32(a,lo);lo=vminq_f32(lo,b);
        }
        const float low=fminf(0.0f,vminvq_f32(lo)),high=fmaxf(0.0f,vmaxvq_f32(hi));
        const float scale=low==high?1.0f:255.0f/(high-low);
        const float inverse=scale?1.0f/scale:0.0f;
        const float dlow=low*scale,dhigh=high*scale;
        const float err_low=-128.0f+dlow,err_high=127.0f+dhigh;
        const float zero=fminf(fmaxf(err_low+err_high>0?-128.0f-dlow:127.0f-dhigh,-128.0f),127.0f);
        const int32_t zp=static_cast<int32_t>(rintf(zero)),negative=-zp;
        for(int64_t kk=0;kk<k;kk+=4) {
          auto q=vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(load4(row+kk),scale)));
          q=vadd_s16(q,vdup_n_s16(static_cast<int16_t>(zp)));
          q=vmax_s16(q,vdup_n_s16(-128));q=vmin_s16(q,vdup_n_s16(127));
          auto bytes=vqmovn_s16(vcombine_s16(q,vdup_n_s16(0)));
          const uint32_t word=vget_lane_u32(vreinterpret_u32_s8(bytes),0);
          std::memcpy(panel+(kk/4*mr+r)*4,&word,4);
        }
        const uint8_t last_code=panel[((k-1)/4*mr+r)*4+(k-1)%4];
        for(int64_t kk=k;kk<kp;++kk)panel[(kk/4*mr+r)*4+kk%4]=last_code;
        std::memcpy(panel+mr*kp+r*4,&negative,4);
        std::memcpy(panel+mr*(kp+4)+r*4,&inverse,4);
      }
    }
  });
  return result;
}
at::Tensor quant_pack_neon(const at::Tensor& input) {return quant_pack_impl(input,true);}
at::Tensor repack_neon_lhs(const at::Tensor& lhs,int64_t m,int64_t k) {
  check_k(k);
  const int64_t sm=WG(get_mr)(),nm=NG(get_mr)(),kp=ceil_div(k,32)*32;
  TORCH_CHECK(m>=0 && sm%nm==0 && nm==4 && lhs.device().is_cpu() && lhs.scalar_type()==at::kByte &&
    lhs.dim()==1 && lhs.is_contiguous() && lhs.numel()==AP(get_lhs_packed_size)(m,k,sm,4,1),"Invalid SME activation for NEON reorder");
  auto out=at::zeros({static_cast<int64_t>(AP(get_lhs_packed_size)(m,k,nm,8,1))},lhs.options());
  const auto* src=lhs.const_data_ptr<uint8_t>();auto* dst=out.data_ptr<uint8_t>();
  kernel_parallel::parallel_for(0,ceil_div(m,nm),1,[&](int64_t begin,int64_t end) {
    for(int64_t block=begin;block<end;++block) {
      auto* dest=dst+block*nm*(kp+8);
      for(int64_t r=0;r<nm && block*nm+r<m;++r) {
        const int64_t row=block*nm+r,inside=row%sm;
        const auto* source=src+(row/sm)*sm*(kp+8);
        for(int64_t kk=0;kk<kp;kk+=8) {
          std::memcpy(dest+(kk/8*nm+r)*8,source+(kk/4*sm+inside)*4,4);
          std::memcpy(dest+(kk/8*nm+r)*8+4,source+((kk/4+1)*sm+inside)*4,4);
        }
        std::memcpy(dest+nm*kp+r*4,source+sm*kp+inside*4,4);
        std::memcpy(dest+nm*(kp+4)+r*4,source+sm*(kp+4)+inside*4,4);
      }
    }
  });
  return out;
}
at::Tensor matmul_impl(const at::Tensor& lhs,const at::Tensor& rhs,int64_t m,int64_t n,int64_t k,int64_t workers,int64_t axis,int64_t mt=0,int64_t nt=0,bool dynamic=false) {
  check_k(k);
  TORCH_CHECK(m>=0 && n>0 && workers>0 && workers<=kernel_parallel::limit() && (axis==0 || axis==1),"Invalid W8 GEMM dimensions/scheduling");
  TORCH_CHECK(lhs.device().is_cpu() && lhs.scalar_type()==at::kByte && lhs.is_contiguous() && lhs.dim()==1 &&
    lhs.numel()==AP(get_lhs_packed_size)(m,k,WG(get_mr)(),WG(get_kr)(),WG(get_sr)()),"Invalid packed activation");
  TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.is_contiguous() && rhs.dim()==1 &&
    rhs.numel()==WP(get_rhs_packed_size)(n,k,WG(get_nr)(),WG(get_kr)(),WG(get_sr)()),"Invalid packed weight");
  auto output=at::empty({m,n},lhs.options().dtype(at::kFloat));
  if(m==0)return output;
  if(mt || nt) {
    TORCH_CHECK(mt>0 && nt>0 && mt%WG(get_m_step)()==0 && nt%WG(get_n_step)()==0,"Invalid W8 GEMM tile");
    const int64_t rows=ceil_div(m,mt),cols=ceil_div(n,nt),total=rows*cols;
    std::atomic<int64_t> next{0};
    auto run_tile=[&](int64_t task) {
      const int64_t row=(task%rows)*mt,col=(task/rows)*nt;
      WG(run)(std::min(mt,m-row),std::min(nt,n-col),k,
        lhs.const_data_ptr<uint8_t>()+WG(get_lhs_packed_offset)(row,k),
        rhs.const_data_ptr<uint8_t>()+WG(get_rhs_packed_offset)(col,k),
        output.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
    };
    kernel_parallel::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
      for(int64_t slot=begin;slot<end;++slot) {
        if(dynamic) {
          int64_t task;
          while((task=next.fetch_add(1,std::memory_order_relaxed))<total)run_tile(task);
        } else {
          const int64_t chunk=ceil_div(total,workers);
          for(int64_t task=slot*chunk;task<std::min((slot+1)*chunk,total);++task)run_tile(task);
        }
      }
    });
    return output;
  }
  const int64_t step=axis==0?WG(get_m_step)():WG(get_n_step)(),extent=axis==0?m:n;
  const int64_t tasks=std::min(workers,ceil_div(extent,step)),tile=ceil_div(ceil_div(extent,step),tasks)*step;
  kernel_parallel::parallel_for(0,tasks,1,[&](int64_t begin,int64_t end) {
    for(int64_t task=begin;task<end;++task) {
      const int64_t start=task*tile;
      if(start>=extent)continue;
      const int64_t row=axis==0?start:0,col=axis==1?start:0,count=std::min(tile,extent-start);
      WG(run)(axis==0?count:m,axis==1?count:n,k,
        lhs.const_data_ptr<uint8_t>()+WG(get_lhs_packed_offset)(row,k),
        rhs.const_data_ptr<uint8_t>()+WG(get_rhs_packed_offset)(col,k),
        output.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
    }
  });
  return output;
}
at::Tensor matmul(const at::Tensor& lhs,const at::Tensor& rhs,int64_t m,int64_t n,int64_t k,int64_t workers,int64_t axis) {
  return matmul_impl(lhs,rhs,m,n,k,workers,axis);
}
at::Tensor linear(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t workers,int64_t axis) {
  auto lhs=quant_pack(input);
  return matmul(lhs,rhs,input.size(0),n,input.size(1),workers,axis);
}
at::Tensor linear_tiled(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t workers,int64_t mt,int64_t nt,bool dynamic) {
  TORCH_CHECK(mt>0 && nt>0,"Tiled GEMM requires positive tile sizes");
  auto lhs=quant_pack(input);
  return matmul_impl(lhs,rhs,input.size(0),n,input.size(1),workers,0,mt,nt,dynamic);
}
at::Tensor linear_bf16(const at::Tensor& input,const at::Tensor& rhs,int64_t n,int64_t workers,int64_t axis,bool dynamic) {
  auto lhs=quant_pack_bf16(input);
  const int64_t mt=dynamic && input.size(0)>=128 && n>=256?(input.size(1)>4096?32:64):0;
  return matmul_impl(lhs,rhs,input.size(0),n,input.size(1),workers,axis,mt,mt?256:0,mt!=0);
}
// Two projections of the same activation: one quantization and one GEMM team.
// Keep each original packed RHS and independent full-K dot products unchanged.
std::tuple<at::Tensor,at::Tensor> linear_pair_bf16(const at::Tensor& input,const at::Tensor& rhs0,const at::Tensor& rhs1,int64_t n,int64_t workers) {
  auto lhs=quant_pack_bf16(input);
  const int64_t m=input.size(0),k=input.size(1);
  TORCH_CHECK(n>0 && workers>0 && workers<=kernel_parallel::limit(),"Invalid paired GEMM dimensions/workers");
  const int64_t size=WP(get_rhs_packed_size)(n,k,WG(get_nr)(),WG(get_kr)(),WG(get_sr)());
  for(const auto& rhs:{rhs0,rhs1})TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() && rhs.numel()==size,"Invalid paired packed weight");
  auto out0=at::empty({m,n},input.options().dtype(at::kFloat));auto out1=at::empty_like(out0);
  if(!m)return {out0,out1};
  const int64_t mt=k>4096?32:64,nt=256,rows=ceil_div(m,mt),cols=ceil_div(n,nt),tasks=rows*cols;
  std::atomic<int64_t> next{0};
  const auto* src=lhs.const_data_ptr<uint8_t>();
  const uint8_t* weights[]={rhs0.const_data_ptr<uint8_t>(),rhs1.const_data_ptr<uint8_t>()};
  float* dest[]={out0.data_ptr<float>(),out1.data_ptr<float>()};
  kernel_parallel::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      int64_t task;
      while((task=next.fetch_add(1,std::memory_order_relaxed))<2*tasks) {
        const int64_t branch=task/tasks,index=task%tasks,row=(index%rows)*mt,col=(index/rows)*nt;
        WG(run)(std::min(mt,m-row),std::min(nt,n-col),k,
          src+WG(get_lhs_packed_offset)(row,k),weights[branch]+WG(get_rhs_packed_offset)(col,k),
          dest[branch]+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      }
    }
  });
  return {out0,out1};
}
// Experimental mixed SME/NEON execution. Each N partition has its own packing
// ABI and immutable boundary. No NEON kernel may read the SME-packed range.
at::Tensor linear_hybrid(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                        int64_t ns,int64_t nn,int64_t sw,int64_t nw) {
  TORCH_CHECK(input.device().is_cpu() && input.dim()==2 &&
    (input.scalar_type()==at::kBFloat16 || input.scalar_type()==at::kFloat),"Invalid hybrid input");
  const int64_t m=input.size(0),k=input.size(1),n=ns+nn;
  check_k(k);
  TORCH_CHECK(ns>0 && nn>0 && ns%WG(get_n_step)()==0 && sw>0 && nw>0 && sw+nw<=kernel_parallel::limit(),"Invalid hybrid partitions/workers");
  auto check_rhs=[&](const at::Tensor& rhs,int64_t count,int64_t nr,int64_t kr) {
    TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
      rhs.numel()==WP(get_rhs_packed_size)(count,k,nr,kr,1),"Invalid hybrid packed weight");
  };
  check_rhs(rhs_sme,ns,WG(get_nr)(),WG(get_kr)());
  check_rhs(rhs_neon,nn,NG(get_nr)(),NG(get_kr)());
  auto output=at::empty({m,n},input.options().dtype(at::kFloat));
  if(!m)return output;
  auto x=input.to(at::kFloat).contiguous();
  auto ls=quant_pack_impl(x,false),ln=repack_neon_lhs(ls,m,k);
  const int64_t st=ceil_div(ceil_div(ns,WG(get_n_step)()),sw)*WG(get_n_step)();
  const int64_t nt=ceil_div(ceil_div(nn,NG(get_n_step)()),nw)*NG(get_n_step)();
  kernel_parallel::parallel_for(0,sw+nw,1,[&](int64_t begin,int64_t end) {
    for(int64_t task=begin;task<end;++task) {
      if(task<sw) {
        const int64_t col=task*st;if(col>=ns)continue;
        WG(run)(m,std::min(st,ns-col),k,ls.const_data_ptr<uint8_t>(),
          rhs_sme.const_data_ptr<uint8_t>()+WG(get_rhs_packed_offset)(col,k),
          output.data_ptr<float>()+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      } else {
        const int64_t col=(task-sw)*nt;if(col>=nn)continue;
        NG(run)(m,std::min(nt,nn-col),k,ln.const_data_ptr<uint8_t>(),
          rhs_neon.const_data_ptr<uint8_t>()+NG(get_rhs_packed_offset)(col,k),
          output.data_ptr<float>()+ns+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      }
    }
  });
  return output;
}
// Keep SME/NEON mathematical partitions fixed, but assign full-K output tiles
// dynamically within each worker group. Reuse the exact BF16 A8 quantizer and
// repack its bytes for NEON; there is still only one activation quantization.
template<bool Trace,bool Cooperative=false,bool Policy=false>
at::Tensor linear_hybrid_dynamic_impl(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                                int64_t ns,int64_t nn,int64_t sw,int64_t nw,
                                int64_t smt,int64_t snt,int64_t nmt,int64_t nnt,at::Tensor* trace,
                                const at::Tensor* cpu_clusters=nullptr,int64_t neon_cluster_mask=0xffffffffLL) {
  TORCH_CHECK(input.device().is_cpu() && input.dim()==2 && input.scalar_type()==at::kBFloat16,"Expected BF16 input");
  const int64_t m=input.size(0),k=input.size(1),n=ns+nn;
  check_k(k);
  TORCH_CHECK(ns>0 && nn>0 && ns%WG(get_n_step)()==0 && sw>0 && nw>0 && sw+nw<=kernel_parallel::limit(),"Invalid hybrid partitions/workers");
  TORCH_CHECK(smt>0 && snt>0 && nmt>0 && nnt>0 && smt%WG(get_m_step)()==0 && snt%WG(get_n_step)()==0 &&
              nmt%NG(get_m_step)()==0 && nnt%NG(get_n_step)()==0,"Invalid hybrid tiles");
  auto check_rhs=[&](const at::Tensor& rhs,int64_t count,int64_t nr,int64_t kr) {
    TORCH_CHECK(rhs.device().is_cpu() && rhs.scalar_type()==at::kByte && rhs.dim()==1 && rhs.is_contiguous() &&
      rhs.numel()==WP(get_rhs_packed_size)(count,k,nr,kr,1),"Invalid hybrid packed weight");
  };
  check_rhs(rhs_sme,ns,WG(get_nr)(),WG(get_kr)());
  check_rhs(rhs_neon,nn,NG(get_nr)(),NG(get_kr)());
  const int64_t* cluster_map=nullptr;int64_t cpu_count=0;
  if constexpr(Cooperative) {
    TORCH_CHECK(neon_cluster_mask>=0 && neon_cluster_mask<=0xffffffffLL,"Invalid NEON cluster mask");
    TORCH_CHECK(cpu_clusters && cpu_clusters->device().is_cpu() && cpu_clusters->scalar_type()==at::kLong &&
                cpu_clusters->dim()==1 && cpu_clusters->is_contiguous() && cpu_clusters->numel()>0,"Invalid CPU cluster map");
    cluster_map=cpu_clusters->const_data_ptr<int64_t>();cpu_count=cpu_clusters->numel();
    for(int64_t i=0;i<cpu_count;++i)TORCH_CHECK(cluster_map[i]>=0 && cluster_map[i]<32,"Cluster id outside supported range");
  }
  auto output=at::empty({m,n},input.options().dtype(at::kFloat));
  if constexpr(Trace)*trace=at::empty({0,8},input.options().dtype(at::kLong));
  if(!m)return output;
  auto ls=quant_pack_bf16(input),ln=repack_neon_lhs(ls,m,k);
  const int64_t sr=ceil_div(m,smt),sc=ceil_div(ns,snt),nr=ceil_div(m,nmt),nc=ceil_div(nn,nnt);
  int64_t* records=nullptr;
  if constexpr(Trace) {
    *trace=at::empty({sr*sc+nr*nc,8},input.options().dtype(at::kLong));
    records=trace->data_ptr<int64_t>();
  }
  auto cpu_number=[]() -> int64_t {size_t cpu=0;return pthread_cpu_number_np(&cpu)==0?static_cast<int64_t>(cpu):-1;};
  auto record=[&](int64_t index,int64_t kind,int64_t slot,int64_t row,int64_t col,int64_t before,uint64_t start) {
    if constexpr(Trace) {
      const uint64_t finish=mach_absolute_time();const int64_t after=cpu_number();
      auto* p=records+index*8;p[0]=kind;p[1]=slot;p[2]=row;p[3]=col;
      p[4]=before;p[5]=after;p[6]=static_cast<int64_t>(start);p[7]=static_cast<int64_t>(finish);
    }
  };
  std::atomic<int64_t> next_sme{0},next_neon{0};
  // Separate cache lines avoid false sharing between cluster-local claims.
  struct alignas(128) Claim {std::atomic<int> active{0};};
  Claim claims[32];
  auto run_sme=[&](int64_t task,int64_t slot) {
    const int64_t row=task%sr*smt,col=task/sr*snt;
    int64_t cpu=-1;uint64_t start=0;
    if constexpr(Trace){cpu=cpu_number();start=mach_absolute_time();}
    WG(run)(std::min(smt,m-row),std::min(snt,ns-col),k,
      ls.const_data_ptr<uint8_t>()+WG(get_lhs_packed_offset)(row,k),
      rhs_sme.const_data_ptr<uint8_t>()+WG(get_rhs_packed_offset)(col,k),
      output.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
    record(task,0,slot,row,col,cpu,start);
  };
  auto run_neon=[&](int64_t task,int64_t slot) {
    const int64_t row=task%nr*nmt,col=task/nr*nnt;
    int64_t cpu=-1;uint64_t start=0;
    if constexpr(Trace){cpu=cpu_number();start=mach_absolute_time();}
    NG(run)(std::min(nmt,m-row),std::min(nnt,nn-col),k,
      ln.const_data_ptr<uint8_t>()+NG(get_lhs_packed_offset)(row,k),
      rhs_neon.const_data_ptr<uint8_t>()+NG(get_rhs_packed_offset)(col,k),
      output.data_ptr<float>()+row*n+ns+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
    record(sr*sc+task,1,slot,row,ns+col,cpu,start);
  };
  kernel_parallel::parallel_for(0,sw+nw,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      int64_t task;
      if constexpr(Cooperative) {
        while(true) {
          if(next_sme.load(std::memory_order_relaxed)<sr*sc) {
            const int64_t cpu=cpu_number();
            const int64_t cluster=cpu>=0 && cpu<cpu_count?cluster_map[cpu]:0;
            int expected=0;
            if(claims[cluster].active.compare_exchange_strong(expected,1,std::memory_order_acquire)) {
              task=next_sme.fetch_add(1,std::memory_order_relaxed);
              if(task<sr*sc)run_sme(task,slot);
              claims[cluster].active.store(0,std::memory_order_release);
              if(task<sr*sc)continue;
            }
          }
          bool allow_neon=true;
          if constexpr(Policy) {
            const int64_t cpu=cpu_number();
            const int64_t cluster=cpu>=0 && cpu<cpu_count?cluster_map[cpu]:0;
            allow_neon=(neon_cluster_mask & (1LL<<cluster))!=0;
          }
          if(allow_neon) {
            task=next_neon.fetch_add(1,std::memory_order_relaxed);
            if(task<nr*nc) {run_neon(task,slot);continue;}
          }
          if constexpr(Policy) {
            // If NEON is exhausted or discouraged here, keep all available
            // workers helping the SME tail. This avoids leaving a migrating
            // single claimant to finish a large remaining queue alone.
            task=next_sme.fetch_add(1,std::memory_order_relaxed);
            if(task<sr*sc) {run_sme(task,slot);continue;}
          }
          break;
        }
        continue;
      }
      if(slot<sw) {
        while((task=next_sme.fetch_add(1,std::memory_order_relaxed))<sr*sc) {
          run_sme(task,slot);
        }
      } else {
        while((task=next_neon.fetch_add(1,std::memory_order_relaxed))<nr*nc) {
          run_neon(task,slot);
        }
      }
    }
  });
  if constexpr(Policy) {
    // Cluster preferences cannot guarantee OS placement. Complete remaining
    // NEON tiles without the preference if no eligible worker consumed them.
    if(next_neon.load(std::memory_order_relaxed)<nr*nc) {
      kernel_parallel::parallel_for(0,sw+nw,1,[&](int64_t begin,int64_t end) {
        for(int64_t slot=begin;slot<end;++slot) {
          int64_t task;
          while((task=next_neon.fetch_add(1,std::memory_order_relaxed))<nr*nc)run_neon(task,slot);
        }
      });
    }
  }
  return output;
}
at::Tensor linear_hybrid_dynamic(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                                int64_t ns,int64_t nn,int64_t sw,int64_t nw,
                                int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  return linear_hybrid_dynamic_impl<false>(input,rhs_sme,rhs_neon,ns,nn,sw,nw,smt,snt,nmt,nnt,nullptr);
}
std::tuple<at::Tensor,at::Tensor> linear_hybrid_dynamic_trace(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                                int64_t ns,int64_t nn,int64_t sw,int64_t nw,
                                int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  at::Tensor trace;
  auto out=linear_hybrid_dynamic_impl<true>(input,rhs_sme,rhs_neon,ns,nn,sw,nw,smt,snt,nmt,nnt,&trace);
  return {out,trace};
}
at::Tensor linear_hybrid_cooperative(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                                int64_t ns,int64_t nn,int64_t workers,const at::Tensor& cpu_clusters,
                                int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  TORCH_CHECK(workers>=2,"Cooperative hybrid requires at least two workers");
  return linear_hybrid_dynamic_impl<false,true>(input,rhs_sme,rhs_neon,ns,nn,1,workers-1,smt,snt,nmt,nnt,nullptr,&cpu_clusters);
}
std::tuple<at::Tensor,at::Tensor> linear_hybrid_cooperative_trace(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                                int64_t ns,int64_t nn,int64_t workers,const at::Tensor& cpu_clusters,
                                int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  TORCH_CHECK(workers>=2,"Cooperative hybrid requires at least two workers");
  at::Tensor trace;
  auto out=linear_hybrid_dynamic_impl<true,true>(input,rhs_sme,rhs_neon,ns,nn,1,workers-1,smt,snt,nmt,nnt,&trace,&cpu_clusters);
  return {out,trace};
}
at::Tensor linear_hybrid_policy(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                                int64_t ns,int64_t nn,int64_t workers,const at::Tensor& cpu_clusters,int64_t neon_cluster_mask,
                                int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  TORCH_CHECK(workers>=2,"Cooperative policy requires at least two workers");
  return linear_hybrid_dynamic_impl<false,true,true>(input,rhs_sme,rhs_neon,ns,nn,1,workers-1,smt,snt,nmt,nnt,nullptr,&cpu_clusters,neon_cluster_mask);
}
std::tuple<at::Tensor,at::Tensor> linear_hybrid_policy_trace(const at::Tensor& input,const at::Tensor& rhs_sme,const at::Tensor& rhs_neon,
                                int64_t ns,int64_t nn,int64_t workers,const at::Tensor& cpu_clusters,int64_t neon_cluster_mask,
                                int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  TORCH_CHECK(workers>=2,"Cooperative policy requires at least two workers");
  at::Tensor trace;
  auto output=linear_hybrid_dynamic_impl<true,true,true>(input,rhs_sme,rhs_neon,ns,nn,1,workers-1,smt,snt,nmt,nnt,&trace,&cpu_clusters,neon_cluster_mask);
  return {output,trace};
}
}
TORCH_LIBRARY(qwen21_w8a8_sme,m) {
  m.def("configure_parallel(bool kernel_owned, int workers) -> ()",&kernel_parallel::configure);
  m.def("parallel_info() -> int[]",&kernel_parallel::info);
  m.def("layout() -> int[]",&layout);
  m.def("pack(Tensor codes, Tensor scales) -> Tensor",&pack);
  m.def("pack_neon(Tensor codes, Tensor scales) -> Tensor",&pack_neon);
  m.def("quant_pack(Tensor input) -> Tensor",&quant_pack);
  m.def("quant_pack_bf16(Tensor input) -> Tensor",&quant_pack_bf16);
  m.def("quant_pack_neon(Tensor input) -> Tensor",&quant_pack_neon);
  m.def("repack_neon_lhs(Tensor input, int m, int k) -> Tensor",&repack_neon_lhs);
  m.def("matmul(Tensor lhs, Tensor rhs, int m, int n, int k, int workers, int axis) -> Tensor",&matmul);
  m.def("linear(Tensor input, Tensor packed, int n, int workers, int axis) -> Tensor",&linear);
  m.def("linear_bf16(Tensor input, Tensor packed, int n, int workers, int axis, bool dynamic) -> Tensor",&linear_bf16);
  m.def("linear_pair_bf16(Tensor input, Tensor packed0, Tensor packed1, int n, int workers) -> (Tensor, Tensor)",&linear_pair_bf16);
  m.def("linear_tiled(Tensor input, Tensor packed, int n, int workers, int mt, int nt, bool dynamic) -> Tensor",&linear_tiled);
  m.def("linear_hybrid(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int sme_workers, int neon_workers) -> Tensor",&linear_hybrid);
  m.def("linear_hybrid_dynamic(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int sme_workers, int neon_workers, int sme_mt, int sme_nt, int neon_mt, int neon_nt) -> Tensor",&linear_hybrid_dynamic);
  m.def("linear_hybrid_dynamic_trace(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int sme_workers, int neon_workers, int sme_mt, int sme_nt, int neon_mt, int neon_nt) -> (Tensor, Tensor)",&linear_hybrid_dynamic_trace);
  m.def("linear_hybrid_cooperative(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int workers, Tensor cpu_clusters, int sme_mt, int sme_nt, int neon_mt, int neon_nt) -> Tensor",&linear_hybrid_cooperative);
  m.def("linear_hybrid_cooperative_trace(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int workers, Tensor cpu_clusters, int sme_mt, int sme_nt, int neon_mt, int neon_nt) -> (Tensor, Tensor)",&linear_hybrid_cooperative_trace);
  m.def("linear_hybrid_policy(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int workers, Tensor cpu_clusters, int neon_cluster_mask, int sme_mt, int sme_nt, int neon_mt, int neon_nt) -> Tensor",&linear_hybrid_policy);
  m.def("linear_hybrid_policy_trace(Tensor input, Tensor sme_packed, Tensor neon_packed, int n_sme, int n_neon, int workers, Tensor cpu_clusters, int neon_cluster_mask, int sme_mt, int sme_nt, int neon_mt, int neon_nt) -> (Tensor, Tensor)",&linear_hybrid_policy_trace);
  m.def("trace_timebase() -> int[]",[](){mach_timebase_info_data_t t;mach_timebase_info(&t);return std::vector<int64_t>{t.numer,t.denom};});
}
