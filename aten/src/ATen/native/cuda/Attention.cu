#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace at::native {

TORCH_IMPL_FUNC(tanh_attention_cuda)(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& output) {
  c10::impl::ExcludeDispatchKeyGuard guardCUDA(c10::DispatchKey::AutocastCUDA);

  Tensor kT = k.transpose(-2, -1);
  Tensor x = at::matmul(q, kT);
  Tensor a = at::tanh(x);
  Tensor result = at::matmul(a, v);
  
  output.copy_(result); 
}

}
