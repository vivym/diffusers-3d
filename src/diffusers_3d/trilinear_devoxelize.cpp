#include <torch/library.h>
#include "diffusers_3d/trilinear_devoxelize.h"

namespace diffusers_3d::trilinear_devoxelize {

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize(
    const at::Tensor& coords, const at::Tensor& voxel_features , int64_t voexl_resolution) {
  TORCH_CHECK(coords.is_contiguous(), "coords must be a contiguous tensor");
  TORCH_CHECK(coords.dim() == 3, "coords must be a 3D tensor");

  TORCH_CHECK(voxel_features.is_contiguous(), "voxel_features must be a contiguous tensor");
  TORCH_CHECK(voxel_features.dim() == 5, "voxel_features must be a 5D tensor");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("diffusers_3d::trilinear_devoxelize", "")
                       .typed<decltype(trilinear_devoxelize)>();
  return op.call(coords, voxel_features, voexl_resolution);
}

at::Tensor trilinear_devoxelize_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t voexl_resolution) {
  TORCH_CHECK(grad_outputs.is_contiguous(), "grad_outputs must be a contiguous tensor");
  TORCH_CHECK(grad_outputs.dim() == 3, "grad_outputs must be a 3D tensor");

  TORCH_CHECK(indices.is_contiguous(), "indices must be a contiguous tensor");
  TORCH_CHECK(indices.dim() == 3, "indices must be a 3D tensor");

  TORCH_CHECK(weights.is_contiguous(), "weights must be a contiguous tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("diffusers_3d::trilinear_devoxelize_backward", "")
                       .typed<decltype(trilinear_devoxelize_backward)>();
  return op.call(grad_outputs, indices, weights, voexl_resolution);
}

TORCH_LIBRARY_FRAGMENT(diffusers_3d, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "diffusers_3d::trilinear_devoxelize(Tensor coords, "
      "Tensor voxel_features, int voexl_resolution) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA("diffusers_3d::trilinear_devoxelize_backward("
      "Tensor grad_outputs, Tensor indices, Tensor weights, int voexl_resolution) -> Tensor"));
}

} // namespace diffusers_3d::trilinear_devoxelize
