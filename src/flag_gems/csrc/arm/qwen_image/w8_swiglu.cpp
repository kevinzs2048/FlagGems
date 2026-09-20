// Experimental: two complete-K W8 projections -> exact rounded SwiGLU ->
// BF16 packed LHS, with only worker-local FP32 projection scratch.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <arm_neon.h>
#include <atomic>
#include <cfloat>
#include <pthread.h>
#include <algorithm>
#include <memory>
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#define NG(name) kai_##name##_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm
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
void store_tile(const float* gate,const float* up,const uint16_t* lut,uint16_t* dst,
                int64_t row,int64_t col,int64_t height,int64_t width,
                int64_t mt,int64_t nt,int64_t mr,int64_t kp,int64_t padded_m,int64_t column_end) {
  // Each task owns aligned M panels and an even output-column range.
  // Write padding explicitly; workers never share output elements.
  const int64_t padded_height=std::min(mt,padded_m-row),padded_width=std::min(nt,column_end-col);
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
        store_tile(gate,up,lut,dst,row,col,height,width,mt,nt,mr,kp,padded_m,kp);
      }
    }
  });
  return out;
}
// Keep both projections and the rounded SwiGLU epilogue in each worker's tile.
// The two backends own disjoint N ranges in the same downstream BF16 layout.
at::Tensor run_hybrid(const at::Tensor& lhs,const at::Tensor& neon_lhs,
    const at::Tensor& gate_sme,const at::Tensor& up_sme,
    const at::Tensor& gate_neon,const at::Tensor& up_neon,const at::Tensor& table,
    int64_t m,int64_t ns,int64_t nn,int64_t k,int64_t workers,const at::Tensor& clusters,
    int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  TORCH_CHECK(m>=0 && ns>0 && nn>0 && k>0 && k%8==0 && k<=32768 &&
              workers>0 && workers<=at::get_num_threads(),"Invalid hybrid dimensions/workers");
  const int64_t n=ns+nn,mr=BG(get_mr)(),kp=divup(n,2)*2;
  TORCH_CHECK(ns%WG(get_n_step)()==0 && ns%2==0,"Invalid hybrid N boundary");
  TORCH_CHECK(smt>0 && smt<=256 && nmt>0 && nmt<=256 && snt>0 && snt<=2048 &&
              nnt>0 && nnt<=2048 && smt%mr==0 && nmt%mr==0 &&
              smt%WG(get_m_step)()==0 && snt%WG(get_n_step)()==0 &&
              nmt%NG(get_m_step)()==0 && nnt%NG(get_n_step)()==0,"Invalid hybrid tile shape");
  auto bytes=[](const at::Tensor& t,int64_t size) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==at::kByte && t.dim()==1 &&
                t.is_contiguous() && t.numel()==size,"Invalid hybrid packed buffer");
  };
  bytes(lhs,AP(get_lhs_packed_size)(m,k,WG(get_mr)(),WG(get_kr)(),WG(get_sr)()));
  bytes(neon_lhs,AP(get_lhs_packed_size)(m,k,NG(get_mr)(),NG(get_kr)(),NG(get_sr)()));
  for(const auto& w:{gate_sme,up_sme})
    bytes(w,WP(get_rhs_packed_size)(ns,k,WG(get_nr)(),WG(get_kr)(),WG(get_sr)()));
  for(const auto& w:{gate_neon,up_neon})
    bytes(w,WP(get_rhs_packed_size)(nn,k,NG(get_nr)(),NG(get_kr)(),NG(get_sr)()));
  TORCH_CHECK(table.device().is_cpu() && table.scalar_type()==at::kBFloat16 && table.dim()==1 &&
              table.numel()==65536 && table.is_contiguous(),"Invalid BF16 SiLU table");
  TORCH_CHECK(clusters.device().is_cpu() && clusters.scalar_type()==at::kLong &&
              clusters.dim()==1 && clusters.numel()>0 && clusters.is_contiguous(),"Invalid CPU cluster map");
  const auto* map=clusters.const_data_ptr<int64_t>();
  for(int64_t i=0;i<clusters.numel();++i)TORCH_CHECK(map[i]>=0 && map[i]<32,"Invalid cluster id");
  auto out=at::empty({divup(m,mr)*mr*kp*2},lhs.options());
  if(!m)return out;
  const int64_t tile_size=std::max(smt*snt,nmt*nnt);
  auto scratch=at::empty({workers,2,tile_size},lhs.options().dtype(at::kFloat));
  const auto* lut=reinterpret_cast<const uint16_t*>(table.const_data_ptr<at::BFloat16>());
  auto* dst=reinterpret_cast<uint16_t*>(out.data_ptr<uint8_t>());
  const int64_t sr=divup(m,smt),sc=divup(ns,snt),nr=divup(m,nmt),nc=divup(nn,nnt);
  const int64_t padded_m=divup(m,mr)*mr;
  std::atomic<int64_t> next_sme{0},next_neon{0};
  struct alignas(128) Claim {std::atomic<int> active{0};};
  Claim claims[32];
  auto execute=[&](int64_t task,int64_t slot,bool neon) {
    const int64_t rows=neon?nr:sr,mt=neon?nmt:smt,nt=neon?nnt:snt;
    const int64_t row=task%rows*mt,col=task/rows*nt;
    const int64_t height=std::min(mt,m-row),width=std::min(nt,(neon?nn:ns)-col);
    auto* gate=scratch.data_ptr<float>()+slot*2*tile_size;auto* up=gate+tile_size;
    for(int branch=0;branch<2;++branch) {
      const auto& w=neon?(branch?up_neon:gate_neon):(branch?up_sme:gate_sme);
      if(neon) {
        NG(run)(height,width,k,neon_lhs.const_data_ptr<uint8_t>()+NG(get_lhs_packed_offset)(row,k),
                w.const_data_ptr<uint8_t>()+NG(get_rhs_packed_offset)(col,k),
                branch?up:gate,nt*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      } else {
        WG(run)(height,width,k,lhs.const_data_ptr<uint8_t>()+WG(get_lhs_packed_offset)(row,k),
                w.const_data_ptr<uint8_t>()+WG(get_rhs_packed_offset)(col,k),
                branch?up:gate,nt*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
      }
    }
    store_tile(gate,up,lut,dst,row,col+(neon?ns:0),height,width,mt,nt,mr,kp,padded_m,neon?kp:ns);
  };
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      while(true) {
        if(next_sme.load(std::memory_order_relaxed)<sr*sc) {
          size_t cpu=0;
          const int64_t cluster=pthread_cpu_number_np(&cpu)==0 && cpu<size_t(clusters.numel())?map[cpu]:0;
          int expected=0;
          if(claims[cluster].active.compare_exchange_strong(expected,1,std::memory_order_acquire)) {
            const int64_t task=next_sme.fetch_add(1,std::memory_order_relaxed);
            if(task<sr*sc)execute(task,slot,false);
            claims[cluster].active.store(0,std::memory_order_release);
            if(task<sr*sc)continue;
          }
        }
        const int64_t task=next_neon.fetch_add(1,std::memory_order_relaxed);
        if(task<nr*nc){execute(task,slot,true);continue;}
        break;
      }
    }
  });
  return out;
}
// Experimental row pipeline: NEON produces rounded packed SwiGLU panels;
// SME2 consumes complete panels for BF16 down projection in the same team.
// Every dot product retains its full K reduction. Release/acquire readiness
// prevents a consumer from reading a panel before all its columns are written.
at::Tensor run_wavefront(const at::Tensor& lhs,const at::Tensor& gate,const at::Tensor& up,
    const at::Tensor& down,const at::Tensor& table,int64_t m,int64_t n,int64_t k,int64_t d,
    int64_t workers,const at::Tensor& clusters,int64_t mt,int64_t nt,int64_t dt) {
  TORCH_CHECK(m>=0 && n>0 && d>0 && k>0 && k%8==0 && k<=32768 &&
              workers>0 && workers<=at::get_num_threads(),"Invalid wavefront dimensions/workers");
  const int64_t mr=BG(get_mr)(),nr=BG(get_nr)(),kp=divup(n,2)*2;
  TORCH_CHECK(mt>0 && mt<=256 && mt%mr==0 && mt%NG(get_m_step)()==0 &&
              nt>0 && nt<=2048 && nt%NG(get_n_step)()==0 && nt%2==0 &&
              dt>0 && dt<=2048 && dt%nr==0,"Invalid wavefront tile");
  auto bytes=[](const at::Tensor& t,int64_t size) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==at::kByte && t.dim()==1 &&
                t.is_contiguous() && t.numel()==size,"Invalid wavefront packed buffer");
  };
  bytes(lhs,AP(get_lhs_packed_size)(m,k,NG(get_mr)(),NG(get_kr)(),NG(get_sr)()));
  for(const auto& w:{gate,up})bytes(w,WP(get_rhs_packed_size)(n,k,NG(get_nr)(),NG(get_kr)(),NG(get_sr)()));
  bytes(down,divup(d,nr)*nr*(4+kp*2));
  TORCH_CHECK(table.device().is_cpu() && table.scalar_type()==at::kBFloat16 && table.dim()==1 &&
              table.numel()==65536 && table.is_contiguous(),"Invalid BF16 SiLU table");
  TORCH_CHECK(clusters.device().is_cpu() && clusters.scalar_type()==at::kLong &&
              clusters.dim()==1 && clusters.numel()>0 && clusters.is_contiguous(),"Invalid CPU cluster map");
  const auto* map=clusters.const_data_ptr<int64_t>();
  for(int64_t i=0;i<clusters.numel();++i)TORCH_CHECK(map[i]>=0 && map[i]<32,"Invalid cluster id");
  auto output=at::empty({m,d},lhs.options().dtype(at::kBFloat16));
  if(!m)return output;
  const int64_t rows=divup(m,mt),cols=divup(n,nt),dcols=divup(d,dt),padded_m=divup(m,mr)*mr;
  auto intermediate=at::empty({padded_m*kp*2},lhs.options());
  const int64_t size=mt*std::max(nt,dt);
  auto scratch=at::empty({workers,2,size},lhs.options().dtype(at::kFloat));
  auto* packed=reinterpret_cast<uint16_t*>(intermediate.data_ptr<uint8_t>());
  auto* dst=reinterpret_cast<uint16_t*>(output.data_ptr<at::BFloat16>());
  const auto* lut=reinterpret_cast<const uint16_t*>(table.const_data_ptr<at::BFloat16>());
  std::unique_ptr<std::atomic<int64_t>[]> ready(new std::atomic<int64_t>[rows]);
  std::unique_ptr<std::atomic<int64_t>[]> ready_rows(new std::atomic<int64_t>[rows]);
  for(int64_t r=0;r<rows;++r) {
    ready[r].store(0,std::memory_order_relaxed);
    ready_rows[r].store(0,std::memory_order_relaxed);
  }
  std::atomic<int64_t> next_up{0},next_down{0},published{0};
  struct alignas(128) Claim {std::atomic<int> active{0};};
  Claim claims[32];
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      float* a=scratch.data_ptr<float>()+slot*2*size;float* b=a+size;
      while(true) {
        int64_t task=next_down.load(std::memory_order_relaxed);
        if(task>=rows*dcols)break;
        const int64_t completed_row=ready_rows[task/dcols].load(std::memory_order_acquire)-1;
        if(completed_row>=0) {
          size_t cpu=0;
          const int64_t cluster=pthread_cpu_number_np(&cpu)==0 && cpu<size_t(clusters.numel())?map[cpu]:0;
          int expected=0;
          if(claims[cluster].active.compare_exchange_strong(expected,1,std::memory_order_acquire)) {
            if(next_down.compare_exchange_strong(task,task+1,std::memory_order_relaxed)) {
              const int64_t row=completed_row*mt,col=task%dcols*dt;
              const int64_t height=std::min(mt,m-row),width=std::min(dt,d-col);
              BG(run)(height,width,n,intermediate.const_data_ptr<uint8_t>()+BG(get_lhs_packed_offset)(row,n),
                      down.const_data_ptr<uint8_t>()+BG(get_rhs_packed_offset)(col,n),a,
                      dt*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
              claims[cluster].active.store(0,std::memory_order_release);
              for(int64_t r=0;r<height;++r) {
                int64_t c=0;
                for(;c+4<=width;c+=4)vst1_u16(dst+(row+r)*d+col+c,rounded(vld1q_f32(a+r*dt+c)));
                for(;c<width;++c)dst[(row+r)*d+col+c]=at::BFloat16(a[r*dt+c]).x;
              }
              continue;
            }
            claims[cluster].active.store(0,std::memory_order_release);
          }
        }
        task=next_up.fetch_add(1,std::memory_order_relaxed);
        if(task>=rows*cols)break;
        const int64_t block=task/cols,row=block*mt,col=task%cols*nt;
        const int64_t height=std::min(mt,m-row),width=std::min(nt,n-col);
        const auto* x=lhs.const_data_ptr<uint8_t>()+NG(get_lhs_packed_offset)(row,k);
        for(int branch=0;branch<2;++branch) {
          const auto& w=branch?up:gate;
          NG(run)(height,width,k,x,w.const_data_ptr<uint8_t>()+NG(get_rhs_packed_offset)(col,k),
                  branch?b:a,nt*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
        }
        store_tile(a,b,lut,packed,row,col,height,width,mt,nt,mr,kp,padded_m,kp);
        if(ready[block].fetch_add(1,std::memory_order_acq_rel)==cols-1) {
          const int64_t index=published.fetch_add(1,std::memory_order_relaxed);
          ready_rows[index].store(block+1,std::memory_order_release);
        }
      }
    }
  });
  return output;
}
}
TORCH_LIBRARY(qwen21_w8_swiglu,m) {
  m.def("run(Tensor lhs, Tensor gate, Tensor up, Tensor table, int m, int n, int k, int workers, int mt=64, int nt=256) -> Tensor",&run);
  m.def("run_hybrid(Tensor lhs, Tensor neon_lhs, Tensor gate_sme, Tensor up_sme, Tensor gate_neon, Tensor up_neon, Tensor table, int m, int ns, int nn, int k, int workers, Tensor clusters, int smt=64, int snt=256, int nmt=128, int nnt=128) -> Tensor",&run_hybrid);
  m.def("run_wavefront(Tensor lhs, Tensor gate, Tensor up, Tensor down, Tensor table, int m, int n, int k, int d, int workers, Tensor clusters, int mt=64, int nt=128, int dt=256) -> Tensor",&run_wavefront);
}
