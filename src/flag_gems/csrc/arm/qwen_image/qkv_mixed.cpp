// Independent Q/K (W8A8 NEON) and V (BF16 SME) share one parallel team.
// Every output tile retains its original complete-K accumulation order.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <atomic>
#include <cfloat>
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#define B(name) kai_##name##_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa
#define N(name) kai_##name##_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm
#define L(name) kai_##name##_lhs_quant_pack_qai8dxp_f32
#define R(name) kai_##name##_rhs_pack_nxk_qsi8cxp_qsi8cx_neon

namespace {
int64_t divup(int64_t a,int64_t b) {return (a+b-1)/b;}
std::tuple<at::Tensor,at::Tensor,at::Tensor> matmul(
    const at::Tensor& bf16_lhs,const at::Tensor& a8_lhs,const at::Tensor& q_rhs,
    const at::Tensor& k_rhs,const at::Tensor& v_rhs,int64_t m,int64_t n,int64_t k,
    int64_t workers,int64_t sme_workers,int64_t smt,int64_t snt,int64_t nmt,int64_t nnt) {
  TORCH_CHECK(m>=0 && n>0 && k>0 && k%8==0 && k<=32768,"Invalid QKV dimensions");
  TORCH_CHECK(workers>=2 && workers<=at::get_num_threads() && sme_workers>0 && sme_workers<workers,"Invalid QKV worker policy");
  TORCH_CHECK(smt>0 && smt%B(get_m_step)()==0 && snt>0 && snt%B(get_n_step)()==0 &&
              nmt>0 && nmt%N(get_m_step)()==0 && nnt>0 && nnt%N(get_n_step)()==0,"Invalid QKV tiles");
  auto check=[](const at::Tensor& tensor,int64_t size) {
    TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type()==at::kByte && tensor.dim()==1 &&
                tensor.is_contiguous() && tensor.numel()==size,"Invalid packed QKV input");
  };
  const int64_t mr=B(get_mr)(),nr=B(get_nr)();
  check(bf16_lhs,divup(m,mr)*mr*divup(k,2)*2*2);
  check(v_rhs,divup(n,nr)*nr*(4+divup(k,2)*2*2));
  check(a8_lhs,L(get_lhs_packed_size)(m,k,N(get_mr)(),N(get_kr)(),1));
  check(q_rhs,R(get_rhs_packed_size)(n,k,N(get_nr)(),N(get_kr)(),1));
  check(k_rhs,q_rhs.numel());
  auto q=at::empty({m,n},bf16_lhs.options().dtype(at::kFloat));
  auto key=at::empty_like(q),v=at::empty_like(q);
  if(!m)return {q,key,v};
  const int64_t sr=divup(m,smt),sc=divup(n,snt),nrtiles=divup(m,nmt),nctiles=divup(n,nnt);
  const int64_t one_projection=nrtiles*nctiles;
  std::atomic<int64_t> next_sme{0},next_neon{0};
  auto sme=[&](int64_t task) {
    const int64_t row=task%sr*smt,col=task/sr*snt;
    B(run)(std::min(smt,m-row),std::min(snt,n-col),k,
           bf16_lhs.const_data_ptr<uint8_t>()+B(get_lhs_packed_offset)(row,k),
           v_rhs.const_data_ptr<uint8_t>()+B(get_rhs_packed_offset)(col,k),
           v.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
  };
  auto neon=[&](int64_t task) {
    const bool is_key=task>=one_projection;task%=one_projection;
    const int64_t row=task%nrtiles*nmt,col=task/nrtiles*nnt;
    const auto& rhs=is_key?k_rhs:q_rhs;auto& out=is_key?key:q;
    N(run)(std::min(nmt,m-row),std::min(nnt,n-col),k,
           a8_lhs.const_data_ptr<uint8_t>()+N(get_lhs_packed_offset)(row,k),
           rhs.const_data_ptr<uint8_t>()+N(get_rhs_packed_offset)(col,k),
           out.data_ptr<float>()+row*n+col,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
  };
  at::parallel_for(0,workers,1,[&](int64_t begin,int64_t end) {
    for(int64_t slot=begin;slot<end;++slot) {
      int64_t task;
      if(slot<sme_workers) {
        while((task=next_sme.fetch_add(1,std::memory_order_relaxed))<sr*sc)sme(task);
        while((task=next_neon.fetch_add(1,std::memory_order_relaxed))<2*one_projection)neon(task);
      } else {
        while((task=next_neon.fetch_add(1,std::memory_order_relaxed))<2*one_projection)neon(task);
        while((task=next_sme.fetch_add(1,std::memory_order_relaxed))<sr*sc)sme(task);
      }
    }
  });
  return {q,key,v};
}
}
TORCH_LIBRARY(qwen21_qkv_mixed,m) {
  m.def("matmul(Tensor bf16_lhs, Tensor a8_lhs, Tensor q_rhs, Tensor k_rhs, Tensor v_rhs, int m, int n, int k, int workers, int sme_workers, int sme_mt, int sme_nt, int neon_mt, int neon_nt) -> (Tensor, Tensor, Tensor)",&matmul);
}
