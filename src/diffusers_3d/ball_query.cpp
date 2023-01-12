#include <torch/library.h>
#include "diffusers_3d/ball_query.h"

namespace diffusers_3d::ball_query {

at::Tensor ball_query(
    const at::Tensor& points,
    const at::Tensor& queries,
    double radius,
    int64_t max_samples_per_query) {
  TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
  TORCH_CHECK(queries.is_contiguous(), "queries must be contiguous");

  TORCH_CHECK(points.dim() == 3, "points must be a 3D tensor");
  TORCH_CHECK(queries.dim() == 3, "queries must be a 3D tensor");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("diffusers_3d::ball_query", "")
                       .typed<decltype(ball_query)>();
  return op.call(points, queries, radius, max_samples_per_query);
}

TORCH_LIBRARY_FRAGMENT(diffusers_3d, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "diffusers_3d::ball_query(Tensor points, Tensor queries, "
      "float radius, int max_samples_per_query) -> Tensor"));
}

} // namespace diffusers_3d::ball_query
