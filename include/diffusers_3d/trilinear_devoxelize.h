#pragma once

#include <torch/types.h>

namespace diffusers_3d::trilinear_devoxelize {

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize(
    const at::Tensor& coords, const at::Tensor& voxel_features , int64_t voexl_resolution);

at::Tensor trilinear_devoxelize_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t voexl_resolution);

} // namespace diffusers_3d::trilinear_devoxelize
