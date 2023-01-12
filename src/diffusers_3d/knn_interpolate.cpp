#include <torch/library.h>
#include "diffusers_3d/knn_interpolate.h"

namespace diffusers_3d::knn_interpolate {

std::tuple<at::Tensor, at::Tensor, at::Tensor> three_nn_interpolate(
    const at::Tensor& src_xyz, const at::Tensor& src_features , const at::Tensor& tgt_xyz) {
  TORCH_CHECK(src_xyz.is_contiguous(), "src_xyz must be a contiguous tensor");
  TORCH_CHECK(src_xyz.dim() == 3, "src_xyz must be a 3D tensor");

  TORCH_CHECK(src_xyz.is_contiguous(), "src_xyz must be a contiguous tensor");
  TORCH_CHECK(src_xyz.dim() == 3, "src_xyz must be a 3D tensor");

  TORCH_CHECK(tgt_xyz.is_contiguous(), "tgt_xyz must be a contiguous tensor");
  TORCH_CHECK(tgt_xyz.dim() == 3, "tgt_xyz must be a 3D tensor");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("diffusers_3d::three_nn_interpolate", "")
                       .typed<decltype(three_nn_interpolate)>();
  return op.call(src_xyz, src_features, tgt_xyz);
}

at::Tensor three_nn_interpolate_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_src_points) {
  TORCH_CHECK(grad_outputs.is_contiguous(), "grad_outputs must be a contiguous tensor");
  TORCH_CHECK(grad_outputs.dim() == 3, "grad_outputs must be a 3D tensor");

  TORCH_CHECK(indices.is_contiguous(), "indices must be a contiguous tensor");
  TORCH_CHECK(indices.dim() == 3, "indices must be a 3D tensor");

  TORCH_CHECK(weights.is_contiguous(), "weights must be a contiguous tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("diffusers_3d::three_nn_interpolate_backward", "")
                       .typed<decltype(three_nn_interpolate_backward)>();
  return op.call(grad_outputs, indices, weights, num_src_points);
}

TORCH_LIBRARY_FRAGMENT(diffusers_3d, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "diffusers_3d::three_nn_interpolate(Tensor src_xyz, "
      "Tensor src_features, Tensor tgt_xyz) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA("diffusers_3d::three_nn_interpolate_backward("
      "Tensor grad_outputs, Tensor indices, Tensor weights, int num_src_points) -> Tensor"));
}

} // namespace diffusers_3d::knn_interpolate
