#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "diffusers_3d/ball_query.h"
#include "diffusers_3d/cuda_utils.h"
#include "diffusers_3d/thrust_allocator.h"

namespace diffusers_3d::ball_query {

template <typename scalar_t, typename index_t, typename policy_t>
void ball_query_cuda_impl_thrust(
    const policy_t& policy,
    index_t* __restrict__ indices_ptr,
    const scalar_t* const __restrict__ points_ptr,
    const scalar_t* const __restrict__ queries_ptr,
    scalar_t radius2,
    index_t batch_size,
    index_t num_points,
    index_t num_queries,
    index_t max_samples_per_query) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(batch_size * num_queries),
      [=] __host__ __device__ (index_t i) {
        const index_t batch_idx = i / num_queries;
        const index_t begin = batch_idx * num_points;
        const index_t end = (batch_idx + 1) * num_points;

        const auto q_x = queries_ptr[i * 3 + 0];
        const auto q_y = queries_ptr[i * 3 + 1];
        const auto q_z = queries_ptr[i * 3 + 2];

        index_t cnt = 0;
        for (auto k = 0; k < num_points && cnt < max_samples_per_query; k++) {
          const auto x = points_ptr[k * 3 + 0];
          const auto y = points_ptr[k * 3 + 1];
          const auto z = points_ptr[k * 3 + 2];
          const auto d2 = (q_x - x) * (q_x - x) + (q_y - y) * (q_y - y) +
                          (q_z - z) * (q_z - z);
          if (d2 < radius2) {
            indices_ptr[i * max_samples_per_query + cnt] = k;
            cnt++;
          }
        }
        for (auto k = cnt; k < max_samples_per_query; k++) {
          indices_ptr[i * max_samples_per_query + k] = indices_ptr[i * max_samples_per_query + 0];
        }
      });
}

template <typename scalar_t, typename index_t>
void ball_query_cuda_impl(
    at::Tensor& indices,
    const at::Tensor& points,
    const at::Tensor& queries,
    double radius,
    int64_t max_samples_per_query) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto batch_size = points.size(0);
  auto num_points = points.size(1);
  auto num_queries = queries.size(1);

  auto indices_ptr = indices.data_ptr<index_t>();
  auto points_ptr = points.data_ptr<scalar_t>();
  auto queries_ptr = queries.data_ptr<scalar_t>();

  ball_query_cuda_impl_thrust<scalar_t, index_t>(
      policy,
      indices_ptr,
      points_ptr,
      queries_ptr,
      static_cast<scalar_t>(radius * radius),
      static_cast<index_t>(batch_size),
      static_cast<index_t>(num_points),
      static_cast<index_t>(num_queries),
      static_cast<index_t>(max_samples_per_query));
}

at::Tensor ball_query_cuda(
    const at::Tensor& points,
    const at::Tensor& queries,
    double radius,
    int64_t max_samples_per_query) {
  TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
  TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");

  auto indices_options = points.options().dtype(at::ScalarType::Long);
  auto indices = at::empty(
      {queries.size(0), queries.size(1), max_samples_per_query}, indices_options);

  AT_DISPATCH_FLOATING_TYPES(points.type(), "ball_query_cuda", [&] {
    ball_query_cuda_impl<scalar_t, int64_t>(
        indices,
        points,
        queries,
        radius,
        max_samples_per_query);
  });

  return indices;
}

TORCH_LIBRARY_IMPL(diffusers_3d, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::ball_query"),
         TORCH_FN(ball_query_cuda));
}

} // namespace diffusers_3d::ball_query
