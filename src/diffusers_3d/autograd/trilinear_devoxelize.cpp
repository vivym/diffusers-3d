#include <torch/autograd.h>

#include "diffusers_3d/trilinear_devoxelize.h"

namespace diffusers_3d::trilinear_devoxelize {

class TrilinearDevoxelizeFunction : public torch::autograd::Function<TrilinearDevoxelizeFunction> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& coords,
      const torch::autograd::Variable& voxel_features,
      int64_t voxel_resolution) {
    ctx->saved_data["voxel_resolution"] = voxel_resolution;

    at::AutoDispatchBelowADInplaceOrView g;
    auto [point_features, indices, weights] = trilinear_devoxelize(
      coords, voxel_features, voxel_resolution);

    ctx->save_for_backward({indices, weights});

    return {point_features, indices, weights};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {
    auto grad_output = grad_outputs[0].contiguous();

    const auto& saved_tensors = ctx->get_saved_variables();
    auto indices = saved_tensors[0];
    auto weights = saved_tensors[1];

    auto voxel_resolution = ctx->saved_data["voxel_resolution"].toInt();

    auto grad_inputs = trilinear_devoxelize_backward(
        grad_output, indices, weights, voxel_resolution);

    return {at::Tensor(), grad_inputs, at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_autograd(
    const at::Tensor& coords, const at::Tensor& voxel_features, int64_t voxel_resolution) {
  auto results = TrilinearDevoxelizeFunction::apply(coords, voxel_features, voxel_resolution);
  return {results[0], results[1], results[2]};
}

TORCH_LIBRARY_IMPL(diffusers_3d, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::trilinear_devoxelize"),
         TORCH_FN(trilinear_devoxelize_autograd));
}

} // namespace diffusers_3d::trilinear_devoxelize
