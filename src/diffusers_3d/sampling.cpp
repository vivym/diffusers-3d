#include <torch/library.h>
#include "diffusers_3d/sampling.h"

namespace diffusers_3d::sampling {

at::Tensor furthest_point_sampling(
    const at::Tensor& coords, int64_t num_samples) {
  TORCH_CHECK(coords.is_contiguous(), "coords must be a contiguous tensor");
  TORCH_CHECK(coords.dim() == 3, "coords must be a 3D tensor");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("diffusers_3d::furthest_point_sampling", "")
                       .typed<decltype(furthest_point_sampling)>();
  return op.call(coords, num_samples);
}

TORCH_LIBRARY_FRAGMENT(diffusers_3d, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "diffusers_3d::furthest_point_sampling(Tensor coords, int num_samples) -> Tensor"));
}

} // namespace diffusers_3d::sampling
