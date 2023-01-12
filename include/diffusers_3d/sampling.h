#pragma once

#include <torch/types.h>

namespace diffusers_3d::sampling {

at::Tensor furthest_point_sampling(const at::Tensor& coords, int64_t num_samples);

} // namespace diffusers_3d::sampling
