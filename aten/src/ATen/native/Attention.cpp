#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace at::native {

Tensor tanh_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
  c10::impl::ExcludeDispatchKeyGuard guardCPU(c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard guardCUDA(c10::DispatchKey::AutocastCUDA);

  Tensor kT = k.transpose(-2, -1);
  Tensor x = at::matmul(q, kT);
  Tensor a = at::tanh(x);
  Tensor result = at::matmul(a, v);
  
  return result;
}

Tensor tanh_attention_meta(const Tensor& q, const Tensor& k, const Tensor& v) {
  c10::impl::ExcludeDispatchKeyGuard guardCPU(c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard guardCUDA(c10::DispatchKey::AutocastCUDA);

  Tensor kT = k.transpose(-2, -1);
  Tensor x = at::matmul(q, kT);
  Tensor a = at::tanh(x);
  Tensor result = at::matmul(a, v);

  return result;
}

}
