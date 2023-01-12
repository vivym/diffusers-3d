#pragma once

#include <torch/types.h>

namespace diffusers_3d::ball_query {

at::Tensor ball_query(
    const at::Tensor& points,
    const at::Tensor& queries,
    double radius,
    int64_t max_samples_per_query);

} // namespace diffusers_3d::ball_query
