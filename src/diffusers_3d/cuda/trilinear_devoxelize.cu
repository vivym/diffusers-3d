#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "diffusers_3d/trilinear_devoxelize.h"
#include "diffusers_3d/cuda_utils.h"
#include "diffusers_3d/thrust_allocator.h"

namespace diffusers_3d::trilinear_devoxelize {

template <typename scalar_t, typename index_t, typename policy_t>
void trilinear_devoxelize_thrust(
    const policy_t& policy,
    scalar_t* __restrict__ outputs_ptr,
    index_t* __restrict__ indices_ptr,
    scalar_t* __restrict__ weights_ptr,
    const scalar_t* const __restrict__ coords_ptr,
    const scalar_t* const __restrict__ voxel_features_ptr,
    int64_t voexl_resolution,
    int64_t batch_size,
    int64_t num_points,
    int64_t num_channels) {
  int64_t r = voexl_resolution;
  int64_t r2 = r * r;
  int64_t r3 = r * r * r;

  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(batch_size * num_points),
      [=] __host__ __device__ (index_t i) {
        const index_t batch_idx = i / num_points;
        const index_t point_idx = i % num_points;

        const auto x = coords_ptr[i * 3 + 0];
        const auto y = coords_ptr[i * 3 + 1];
        const auto z = coords_ptr[i * 3 + 2];

        const auto x_lo_f = floorf(x);
        const auto y_lo_f = floorf(y);
        const auto z_lo_f = floorf(z);

        const auto x_d_1 = x - x_lo_f;
        const auto y_d_1 = y - y_lo_f;
        const auto z_d_1 = z - z_lo_f;
        const auto x_d_0 = 1.0f - x_d_1;
        const auto y_d_0 = 1.0f - y_d_1;
        const auto z_d_0 = 1.0f - z_d_1;

        const auto wgt000 = x_d_0 * y_d_0 * z_d_0;
        const auto wgt001 = x_d_0 * y_d_0 * z_d_1;
        const auto wgt010 = x_d_0 * y_d_1 * z_d_0;
        const auto wgt011 = x_d_0 * y_d_1 * z_d_1;
        const auto wgt100 = x_d_1 * y_d_0 * z_d_0;
        const auto wgt101 = x_d_1 * y_d_0 * z_d_1;
        const auto wgt110 = x_d_1 * y_d_1 * z_d_0;
        const auto wgt111 = x_d_1 * y_d_1 * z_d_1;

        index_t x_lo = static_cast<index_t>(x_lo_f);
        index_t y_lo = static_cast<index_t>(y_lo_f);
        index_t z_lo = static_cast<index_t>(z_lo_f);
        index_t x_hi = (x_d_1 > 0) ? -1 : 0;
        index_t y_hi = (y_d_1 > 0) ? -1 : 0;
        index_t z_hi = (z_d_1 > 0) ? 1 : 0;

        index_t idx000 = x_lo * r2 + y_lo * r + z_lo;
        index_t idx001 = idx000 + z_hi;         // x_lo * r2 + y_lo * r + z_hi;
        index_t idx010 = idx000 + (y_hi & r);   // x_lo * r2 + y_hi * r + z_lo;
        index_t idx011 = idx010 + z_hi;         // x_lo * r2 + y_hi * r + z_hi;
        index_t idx100 = idx000 + (x_hi & r2);  // x_hi * r2 + y_lo * r + z_lo;
        index_t idx101 = idx100 + z_hi;         // x_hi * r2 + y_lo * r + z_hi;
        index_t idx110 = idx100 + (y_hi & r);   // x_hi * r2 + y_hi * r + z_lo;
        index_t idx111 = idx110 + z_hi;         // x_hi * r2 + y_hi * r + z_hi;

        weights_ptr[i * 8 + 0] = wgt000;
        weights_ptr[i * 8 + 1] = wgt001;
        weights_ptr[i * 8 + 2] = wgt010;
        weights_ptr[i * 8 + 3] = wgt011;
        weights_ptr[i * 8 + 4] = wgt100;
        weights_ptr[i * 8 + 5] = wgt101;
        weights_ptr[i * 8 + 6] = wgt110;
        weights_ptr[i * 8 + 7] = wgt111;

        indices_ptr[i * 8 + 0] = idx000;
        indices_ptr[i * 8 + 1] = idx001;
        indices_ptr[i * 8 + 2] = idx010;
        indices_ptr[i * 8 + 3] = idx011;
        indices_ptr[i * 8 + 4] = idx100;
        indices_ptr[i * 8 + 5] = idx101;
        indices_ptr[i * 8 + 6] = idx110;
        indices_ptr[i * 8 + 7] = idx111;

        for (index_t j = 0; j < num_channels; j++) {
          outputs_ptr[batch_idx * num_channels * num_points + j * num_points + point_idx] =
              wgt000 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx000] +
              wgt001 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx001] +
              wgt010 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx010] +
              wgt011 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx011] +
              wgt100 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx100] +
              wgt101 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx101] +
              wgt110 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx110] +
              wgt111 * voxel_features_ptr[batch_idx * num_channels * r3 + j * r3 + idx111];
        }
      });
}

template <typename scalar_t, typename index_t>
void trilinear_devoxelize_cuda_impl(
    at::Tensor& outputs,
    at::Tensor& indices,
    at::Tensor& weights,
    const at::Tensor& coords,
    const at::Tensor& voxel_features,
    int64_t voexl_resolution) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto batch_size = coords.size(0);
  auto num_points = coords.size(1);
  auto num_channels = voxel_features.size(1);

  auto outputs_ptr = outputs.data_ptr<scalar_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto weights_ptr = weights.data_ptr<scalar_t>();
  auto coords_ptr = coords.data_ptr<scalar_t>();
  auto voxel_features_ptr = voxel_features.data_ptr<scalar_t>();

  trilinear_devoxelize_thrust<scalar_t, index_t>(
      policy, outputs_ptr, indices_ptr, weights_ptr, coords_ptr, voxel_features_ptr,
      voexl_resolution, batch_size, num_points, num_channels);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_cuda(
    const at::Tensor& coords, const at::Tensor& voxel_features , int64_t voexl_resolution) {
  TORCH_CHECK(coords.is_cuda(), "coords must be a CUDA tensor");
  TORCH_CHECK(voxel_features.is_cuda(), "voxel_features must be a CUDA tensor");

  auto batch_size = coords.size(0);
  auto num_points = coords.size(1);
  auto num_channels = voxel_features.size(1);

  auto outputs = at::empty({batch_size, num_channels, num_points}, voxel_features.options());
  auto indices_options = coords.options().dtype(at::ScalarType::Long);
  auto indices = at::empty({batch_size, num_points, 8}, indices_options);
  auto weights = at::empty({batch_size, num_points, 8}, voxel_features.options());

  AT_DISPATCH_FLOATING_TYPES(coords.type(), "trilinear_devoxelize_cuda", [&] {
    trilinear_devoxelize_cuda_impl<scalar_t, int64_t>(
        outputs, indices, weights, coords, voxel_features, voexl_resolution);
  });

  return {outputs, indices, weights};
}

template <typename scalar_t, typename index_t, typename policy_t>
void trilinear_devoxelize_backward_thrust(
    const policy_t& policy,
    scalar_t* __restrict__ grad_inputs_ptr,
    const scalar_t* const __restrict__ grad_outputs_ptr,
    const index_t* const __restrict__ indices_ptr,
    const scalar_t* const __restrict__ weights_ptr,
    int64_t voexl_resolution,
    int64_t batch_size,
    int64_t num_points,
    int64_t num_channels) {
  int64_t r3 = voexl_resolution * voexl_resolution * voexl_resolution;

  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(batch_size * num_channels * num_points),
      [=] __host__ __device__ (index_t i) {
        const index_t batch_idx = i / (num_channels * num_points);
        const index_t point_idx = i % num_points;

        const auto indice_offset = batch_idx * num_points * 8 + point_idx * 8;
        const auto idx000 = indices_ptr[indice_offset + 0];
        const auto idx001 = indices_ptr[indice_offset + 1];
        const auto idx010 = indices_ptr[indice_offset + 2];
        const auto idx011 = indices_ptr[indice_offset + 3];
        const auto idx100 = indices_ptr[indice_offset + 4];
        const auto idx101 = indices_ptr[indice_offset + 5];
        const auto idx110 = indices_ptr[indice_offset + 6];
        const auto idx111 = indices_ptr[indice_offset + 7];

        const auto wgt000 = weights_ptr[indice_offset + 0];
        const auto wgt001 = weights_ptr[indice_offset + 1];
        const auto wgt010 = weights_ptr[indice_offset + 2];
        const auto wgt011 = weights_ptr[indice_offset + 3];
        const auto wgt100 = weights_ptr[indice_offset + 4];
        const auto wgt101 = weights_ptr[indice_offset + 5];
        const auto wgt110 = weights_ptr[indice_offset + 6];
        const auto wgt111 = weights_ptr[indice_offset + 7];

        const auto grad_output = grad_outputs_ptr[i];
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx000, grad_output * wgt000);
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx001, grad_output * wgt001);
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx010, grad_output * wgt010);
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx011, grad_output * wgt011);
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx100, grad_output * wgt100);
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx101, grad_output * wgt101);
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx110, grad_output * wgt110);
        atomicAdd(grad_inputs_ptr + i / num_points * r3 + idx111, grad_output * wgt111);
      });
}

template <typename scalar_t, typename index_t>
void trilinear_devoxelize_backward_cuda_impl(
    at::Tensor& grad_inputs,
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t voexl_resolution) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto batch_size = grad_outputs.size(0);
  auto num_channels = grad_outputs.size(1);
  auto num_points = grad_outputs.size(2);

  auto grad_inputs_ptr = grad_inputs.data_ptr<scalar_t>();
  auto grad_outputs_ptr = grad_outputs.data_ptr<scalar_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto weights_ptr = weights.data_ptr<scalar_t>();

  trilinear_devoxelize_backward_thrust<scalar_t, index_t>(
      policy, grad_inputs_ptr, grad_outputs_ptr, indices_ptr, weights_ptr,
      voexl_resolution, batch_size, num_points, num_channels);
}

at::Tensor trilinear_devoxelize_backward_cuda(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t voexl_resolution) {
  TORCH_CHECK(grad_outputs.is_cuda(), "grad_outputs must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

  auto batch_size = grad_outputs.size(0);
  auto num_channels = grad_outputs.size(1);

  auto grad_inputs = at::zeros(
      {batch_size, num_channels, voexl_resolution, voexl_resolution, voexl_resolution},
      grad_outputs.options());

  AT_DISPATCH_FLOATING_TYPES(grad_outputs.type(), "trilinear_devoxelize_backward_cuda", [&] {
    trilinear_devoxelize_backward_cuda_impl<scalar_t, int64_t>(
        grad_inputs, grad_outputs, indices, weights, voexl_resolution);
  });

  return grad_inputs;
}

TORCH_LIBRARY_IMPL(diffusers_3d, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::trilinear_devoxelize"),
         TORCH_FN(trilinear_devoxelize_cuda));
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::trilinear_devoxelize_backward"),
         TORCH_FN(trilinear_devoxelize_backward_cuda));
}

} // namespace diffusers_3d::trilinear_devoxelize
