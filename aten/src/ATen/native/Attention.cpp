#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include <ATen/ops/tanh_attention.h>
#include <ATen/ops/tanh_attention_meta.h>
#include <ATen/ops/tanh_attention_native.h>

namespace at::meta {

TORCH_META_FUNC(tanh_attention)(const Tensor& q, const Tensor& k, const Tensor& v) {
  TORCH_CHECK(q.dim() >= 2 && k.dim() >= 2 && v.dim() >= 2, "q, k, v must be >=2D");
  TORCH_CHECK(q.size(-1) == k.size(-1), "q.size(-1) must equal k.size(-1)");
  TORCH_CHECK(k.size(-2) == v.size(-2), "k.size(-2) must equal v.size(-2)");

  c10::impl::ExcludeDispatchKeyGuard guardCPU(c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard guardCUDA(c10::DispatchKey::AutocastCUDA);

  Tensor kT = k.transpose(-2, -1);
  Tensor x = at::matmul(q, kT);
  Tensor a = at::tanh(x);
  Tensor result = at::matmul(a, v);

  auto& output = maybe_get_output(0);
  set_output_raw_strided(0, result.sizes(), result.strides(), q.options());
  TORCH_CHECK(result.sizes() == output.sizes(), "Expected an output tensor with shape [", result.sizes(), "], but got [", output.sizes(), "] instead.");
}

}

namespace at::native {

TORCH_IMPL_FUNC(tanh_attention_cpu)(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& output) {
  c10::impl::ExcludeDispatchKeyGuard guardCPU(c10::DispatchKey::AutocastCPU);

  Tensor kT = k.transpose(-2, -1);
  Tensor x = at::matmul(q, kT);
  Tensor a = at::tanh(x);
  Tensor result = at::matmul(a, v);
  
  output.copy_(result); 
}

}
