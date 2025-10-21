#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <torch/library.h>

namespace at::native {

// CompositeImplicitAutograd implementation
Tensor tanh_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
  Tensor x = at::matmul(q, k.transpose(-2, -1));
  Tensor a = at::tanh(x);
  return at::matmul(a, v);
}

Tensor tanh_attention_meta(const Tensor& q, const Tensor& k, const Tensor& v) {
  // Broadcast the leading batch dimensions
  auto bq = q.sizes().slice(0, q.dim() - 2);
  auto bk = k.sizes().slice(0, k.dim() - 2);
  auto bv = v.sizes().slice(0, v.dim() - 2);
  auto bqk = infer_size(bq, bk);
  auto batch_shape = infer_size(IntArrayRef(bqk), bv);

  // Final output shape: (..., Lq, Ev)
  DimVector out_sizes;
  out_sizes.insert(out_sizes.end(), batch_shape.begin(), batch_shape.end());
  out_sizes.push_back(q.size(-2));
  out_sizes.push_back(v.size(-1));

  // Return an empty meta tensor (no data allocation)
  return at::empty(out_sizes, q.options());

}

}
