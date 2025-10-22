#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <torch/library.h>

namespace at::native {

Tensor tanh_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
  Tensor x = at::matmul(q, k.transpose(-2, -1));
  Tensor a = at::tanh(x);
  return at::matmul(a, v);
}

Tensor tanh_attention_meta(const Tensor& q, const Tensor& k, const Tensor& v) {
  auto x = at::matmul(q, k.transpose(-2, -1));
  auto a = at::tanh(x);
  return at::matmul(a, v);
}

}
