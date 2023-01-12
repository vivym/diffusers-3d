#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "diffusers_3d/knn_interpolate.h"
#include "diffusers_3d/cuda_utils.h"
#include "diffusers_3d/thrust_allocator.h"

namespace diffusers_3d::knn_interpolate {

template <typename scalar_t, typename index_t, typename policy_t>
void three_nn_thrust(
    const policy_t& policy,
    index_t* __restrict__ indices_ptr,
    scalar_t* __restrict__ weights_ptr,
    const scalar_t* const __restrict__ src_xyz_ptr,
    const scalar_t* const __restrict__ tgt_xyz_ptr,
    int64_t batch_size,
    int64_t num_src_points,
    int64_t num_tgt_points) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(batch_size * num_tgt_points),
      [=] __host__ __device__ (index_t i) {
        const index_t batch_idx = i / num_tgt_points;

        const auto x = tgt_xyz_ptr[i * 3 + 0];
        const auto y = tgt_xyz_ptr[i * 3 + 1];
        const auto z = tgt_xyz_ptr[i * 3 + 2];

        scalar_t best_val0 = 1e10, best_val1 = 1e10, best_val2 = 1e10;
        index_t best_idx0 = 0, best_idx1 = 0, best_idx2 = 0;
        for (int k = 0; k < num_src_points; k++) {
          const auto q_x = src_xyz_ptr[(batch_idx * num_src_points + k) * 3 + 0];
          const auto q_y = src_xyz_ptr[(batch_idx * num_src_points + k) * 3 + 1];
          const auto q_z = src_xyz_ptr[(batch_idx * num_src_points + k) * 3 + 2];

          const auto d2 = (q_x - x) * (q_x - x) + (q_y - y) * (q_y - y) +
                          (q_z - z) * (q_z - z);

          if (d2 < best_val2) {
            best_val2 = d2;
            best_idx2 = k;
            if (d2 < best_val1) {
              best_val2 = best_val1;
              best_idx2 = best_idx1;
              best_val1 = d2;
              best_idx1 = k;
              if (d2 < best_val0) {
              best_val1 = best_val0;
              best_idx1 = best_idx0;
              best_val0 = d2;
              best_idx0 = k;
              }
            }
          }
        }

        best_val0 = max(min(1e10f, best_val0), 1e-10f);
        best_val1 = max(min(1e10f, best_val1), 1e-10f);
        best_val2 = max(min(1e10f, best_val2), 1e-10f);
        const scalar_t d0d1 = best_val0 * best_val1;
        const scalar_t d0d2 = best_val0 * best_val2;
        const scalar_t d1d2 = best_val1 * best_val2;
        const scalar_t d0d1d2 = 1.0f / (d0d1 + d0d2 + d1d2);

        indices_ptr[i * 3 + 0] = best_idx0;
        weights_ptr[i * 3 + 0] = d1d2 * d0d1d2;
        indices_ptr[i * 3 + 1] = best_idx1;
        weights_ptr[i * 3 + 1] = d0d2 * d0d1d2;
        indices_ptr[i * 3 + 0] = best_idx2;
        weights_ptr[i * 3 + 0] = d0d1 * d0d1d2;
      });
}

template <typename scalar_t, typename index_t, typename policy_t>
void three_nn_interpolate_thrust(
    const policy_t& policy,
    scalar_t* __restrict__ outputs_ptr,
    const scalar_t* const __restrict__ src_features_ptr,
    const index_t* const __restrict__ indices_ptr,
    const scalar_t* const __restrict__ weights_ptr,
    int64_t batch_size,
    int64_t num_src_points,
    int64_t num_tgt_points,
    int64_t num_channels) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(batch_size * num_tgt_points * num_channels),
      [=] __host__ __device__ (index_t i) {
        const index_t batch_idx = i / (num_tgt_points * num_channels);
        const index_t channel_idx = i % num_channels;

        const index_t j = i / num_channels;
        const auto idx0 = indices_ptr[j * 3 + 0];
        const auto idx1 = indices_ptr[j * 3 + 1];
        const auto idx2 = indices_ptr[j * 3 + 2];
        const auto weight0 = weights_ptr[j * 3 + 0];
        const auto weight1 = weights_ptr[j * 3 + 1];
        const auto weight2 = weights_ptr[j * 3 + 2];

        const auto offset_0 = (batch_idx * num_src_points + idx0) * num_channels + channel_idx;
        const auto offset_1 = (batch_idx * num_src_points + idx1) * num_channels + channel_idx;
        const auto offset_2 = (batch_idx * num_src_points + idx2) * num_channels + channel_idx;

        outputs_ptr[i] = src_features_ptr[offset_0] * weight0 +
                         src_features_ptr[offset_1] * weight1 +
                         src_features_ptr[offset_2] * weight2;
      });
}

template <typename scalar_t, typename index_t>
void three_nn_interpolate_cuda_impl(
    at::Tensor& outputs,
    at::Tensor& indices,
    at::Tensor& weights,
    const at::Tensor& src_xyz,
    const at::Tensor& src_features,
    const at::Tensor& tgt_xyz) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto batch_size = src_xyz.size(0);
  auto num_src_points = src_xyz.size(1);
  auto num_tgt_points = tgt_xyz.size(1);
  auto num_channels = src_features.size(2);

  auto outputs_ptr = outputs.data_ptr<scalar_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto weights_ptr = weights.data_ptr<scalar_t>();
  auto src_xyz_ptr = src_xyz.data_ptr<scalar_t>();
  auto src_features_ptr = src_features.data_ptr<scalar_t>();
  auto tgt_xyz_ptr = tgt_xyz.data_ptr<scalar_t>();

  three_nn_thrust<scalar_t, index_t>(
      policy, indices_ptr, weights_ptr,
      src_xyz_ptr, tgt_xyz_ptr,
      batch_size, num_src_points, num_tgt_points);

  three_nn_interpolate_thrust<scalar_t, index_t>(
      policy, outputs_ptr, src_features_ptr, indices_ptr, weights_ptr,
      batch_size, num_src_points, num_tgt_points, num_channels);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> three_nn_interpolate_cuda(
    const at::Tensor& src_xyz, const at::Tensor& src_features , const at::Tensor& tgt_xyz) {
  TORCH_CHECK(src_xyz.is_cuda(), "src_xyz must be a CUDA tensor");
  TORCH_CHECK(src_features.is_cuda(), "src_features must be a CUDA tensor");
  TORCH_CHECK(tgt_xyz.is_cuda(), "tgt_xyz must be a CUDA tensor");

  auto batch_size = tgt_xyz.size(0);
  auto num_tgt_points = tgt_xyz.size(1);
  auto num_channels = src_features.size(2);

  auto outputs = at::empty({batch_size, num_tgt_points, num_channels}, src_features.options());
  auto indices_options = src_features.options().dtype(at::ScalarType::Long);
  auto indices = at::empty({batch_size, num_tgt_points, 3}, indices_options);
  auto weights = at::empty({batch_size, num_tgt_points, 3}, src_features.options());

  AT_DISPATCH_FLOATING_TYPES(src_xyz.type(), "three_nn_interpolate_cuda", [&] {
    three_nn_interpolate_cuda_impl<scalar_t, int64_t>(
        outputs, indices, weights, src_xyz, src_features, tgt_xyz);
  });

  return {outputs, indices, weights};
}

template <typename scalar_t, typename index_t, typename policy_t>
void three_nn_interpolate_backward_thrust(
    const policy_t& policy,
    scalar_t* __restrict__ grad_inputs_ptr,
    const scalar_t* const __restrict__ grad_outputs_ptr,
    const index_t* const __restrict__ indices_ptr,
    const scalar_t* const __restrict__ weights_ptr,
    int64_t batch_size,
    int64_t num_src_points,
    int64_t num_tgt_points,
    int64_t num_channels) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(batch_size * num_tgt_points * num_channels),
      [=] __host__ __device__ (index_t i) {
        const index_t batch_idx = i / (num_tgt_points * num_channels);
        const index_t channel_idx = i % num_channels;

        const index_t j = i / num_channels;
        const auto idx0 = indices_ptr[j * 3 + 0];
        const auto idx1 = indices_ptr[j * 3 + 1];
        const auto idx2 = indices_ptr[j * 3 + 2];
        const auto weight0 = weights_ptr[j * 3 + 0];
        const auto weight1 = weights_ptr[j * 3 + 1];
        const auto weight2 = weights_ptr[j * 3 + 2];

        const auto offset_0 = (batch_idx * num_src_points + idx0) * num_channels + channel_idx;
        const auto offset_1 = (batch_idx * num_src_points + idx1) * num_channels + channel_idx;
        const auto offset_2 = (batch_idx * num_src_points + idx2) * num_channels + channel_idx;

        atomicAdd(grad_inputs_ptr + offset_0, grad_outputs_ptr[i] * weight0);
        atomicAdd(grad_inputs_ptr + offset_1, grad_outputs_ptr[i] * weight1);
        atomicAdd(grad_inputs_ptr + offset_2, grad_outputs_ptr[i] * weight2);
      });
}

template <typename scalar_t, typename index_t>
void three_nn_interpolate_backward_cuda_impl(
    at::Tensor& grad_inputs,
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_src_points) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto batch_size = grad_outputs.size(0);
  auto num_tgt_points = grad_outputs.size(1);
  auto num_channels = grad_outputs.size(2);

  auto grad_inputs_ptr = grad_inputs.data_ptr<scalar_t>();
  auto grad_outputs_ptr = grad_outputs.data_ptr<scalar_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto weights_ptr = weights.data_ptr<scalar_t>();

  three_nn_interpolate_backward_thrust<scalar_t, index_t>(
      policy, grad_inputs_ptr, grad_outputs_ptr, indices_ptr, weights_ptr,
      batch_size, num_src_points, num_tgt_points, num_channels);
}

at::Tensor three_nn_interpolate_backward_cuda(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_src_points) {
  TORCH_CHECK(grad_outputs.is_cuda(), "grad_outputs must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

  auto batch_size = grad_outputs.size(0);
  auto num_tgt_points = grad_outputs.size(1);
  auto num_channels = grad_outputs.size(2);

  auto grad_inputs = at::zeros(
      {batch_size, num_src_points, num_channels}, grad_outputs.options());

  AT_DISPATCH_FLOATING_TYPES(grad_outputs.type(), "three_nn_interpolate_backward_cuda", [&] {
    three_nn_interpolate_backward_cuda_impl<scalar_t, int64_t>(
        grad_inputs, grad_outputs, indices, weights, num_src_points);
  });

  return grad_inputs;
}

TORCH_LIBRARY_IMPL(diffusers_3d, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::three_nn_interpolate"),
         TORCH_FN(three_nn_interpolate_cuda));
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::three_nn_interpolate_backward"),
         TORCH_FN(three_nn_interpolate_backward_cuda));
}

} // namespace diffusers_3d::knn_interpolate
