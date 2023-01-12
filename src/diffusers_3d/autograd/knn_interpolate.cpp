#include <torch/autograd.h>

#include "diffusers_3d/knn_interpolate.h"

namespace diffusers_3d::knn_interpolate {

class ThreeNNInterpolateFunction : public torch::autograd::Function<ThreeNNInterpolateFunction> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& src_xyz,
      const torch::autograd::Variable& src_features,
      const torch::autograd::Variable& tgt_xyz) {
    ctx->saved_data["num_src_points"] = src_xyz.size(1);

    at::AutoDispatchBelowADInplaceOrView g;
    auto [tgt_features, indices, weights] = three_nn_interpolate(
      src_xyz, src_features, tgt_xyz);

    ctx->save_for_backward({indices, weights});

    return {tgt_features, indices, weights};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {
    auto grad_output = grad_outputs[0].contiguous();

    const auto& saved_tensors = ctx->get_saved_variables();
    auto indices = saved_tensors[0];
    auto weights = saved_tensors[1];

    auto num_src_points = ctx->saved_data["num_src_points"].toInt();

    auto grad_inputs = three_nn_interpolate_backward(
        grad_output, indices, weights, num_src_points);

    return {grad_inputs, at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> three_nn_interpolate_autograd(
    const at::Tensor& src_xyz, const at::Tensor& src_features, const at::Tensor& tgt_xyz) {
  auto results = ThreeNNInterpolateFunction::apply(src_xyz, src_features, tgt_xyz);
  return {results[0], results[1], results[2]};
}

TORCH_LIBRARY_IMPL(diffusers_3d, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::three_nn_interpolate"),
         TORCH_FN(three_nn_interpolate_autograd));
}

} // namespace diffusers_3d::knn_interpolate
