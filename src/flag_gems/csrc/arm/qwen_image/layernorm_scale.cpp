// Retain ATen's RowwiseMoments implementation and both BF16 rounding points.
// Reference: PyTorch 70d99e998b4955e0049d13a98d77ae1b14db1f45,
// aten/src/ATen/native/cpu/layer_norm_kernel.cpp (BSD-3-Clause).
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/moments_utils.h>
#include <torch/library.h>
#include <cmath>

namespace {
at::Tensor run(const at::Tensor& input,const at::Tensor& modulation,double eps) {
  TORCH_CHECK(input.device().is_cpu() && modulation.device().is_cpu() &&
              input.scalar_type()==at::kBFloat16 && modulation.scalar_type()==at::kBFloat16 &&
              input.dim()==3 && input.size(2)>0,"Expected CPU BF16 [B,S,D]");
  TORCH_CHECK(modulation.dim()==3 && modulation.size(0)==input.size(0) && modulation.size(2)==input.size(2) &&
              (modulation.size(1)==1 || modulation.size(1)==input.size(1)),"Invalid modulation shape");
  auto x=input.contiguous(),s=modulation.contiguous();auto out=at::empty_like(x);
  using B=at::BFloat16;using BV=at::vec::Vectorized<B>;using FV=at::vec::Vectorized<float>;
  const auto* xp=x.const_data_ptr<B>();const auto* sp=s.const_data_ptr<B>();auto* yp=out.data_ptr<B>();
  const int64_t rows=input.size(0)*input.size(1),d=input.size(2),seq=input.size(1);
  const float epsilon=static_cast<float>(eps);
  at::parallel_for(0,rows,1,[&](int64_t begin,int64_t end) {
    for(int64_t row=begin;row<end;++row) {
      const auto* values=xp+row*d;
      const auto* scales=sp+(modulation.size(1)==1?row/seq:row)*d;
      auto [mean,variance]=at::native::RowwiseMoments(values,d);
      const float inverse=1.0f/std::sqrt(variance+epsilon),bias=-inverse*mean;
      int64_t j=0;
      for(;j+BV::size()<=d;j+=BV::size()) {
        auto [a,b]=at::vec::convert_to_float<B>(BV::loadu(values+j));
        auto normalized=at::vec::convert_from_float<B>(
            (a*FV(inverse)+FV(bias))*FV(1)+FV(0),
            (b*FV(inverse)+FV(bias))*FV(1)+FV(0));
        auto [na,nb]=at::vec::convert_to_float<B>(normalized);
        auto [sa,sb]=at::vec::convert_to_float<B>(BV::loadu(scales+j));
        at::vec::convert_from_float<B>(na*sa,nb*sb).store(yp+row*d+j);
      }
      for(;j<d;++j) {
        const B normalized=(float(values[j])*inverse+bias)*1.0f+0.0f;
        yp[row*d+j]=B(float(normalized)*float(scales[j]));
      }
    }
  });
  return out;
}
}
TORCH_LIBRARY(qwen21_layernorm_scale,m) {
  m.def("run(Tensor input, Tensor modulation, float eps) -> Tensor",&run);
  m.def("vector_sizes() -> int[]",[](){return std::vector<int64_t>{at::vec::Vectorized<at::BFloat16>::size(),at::vec::Vectorized<float>::size()};});
}
