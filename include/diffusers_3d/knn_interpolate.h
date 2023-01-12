#pragma once

#include <torch/types.h>

namespace diffusers_3d::knn_interpolate {

std::tuple<at::Tensor, at::Tensor, at::Tensor> three_nn_interpolate(
    const at::Tensor& src_xyz, const at::Tensor& src_features , const at::Tensor& tgt_xyz);

at::Tensor three_nn_interpolate_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_src_points);

} // namespace diffusers_3d::knn_interpolate
