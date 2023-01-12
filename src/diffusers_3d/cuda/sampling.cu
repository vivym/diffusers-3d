#include <cmath>
#include <torch/library.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "diffusers_3d/sampling.h"
#include "diffusers_3d/cuda_utils.h"

namespace diffusers_3d::sampling {

constexpr int kTotalThreads = 512;

inline int opt_num_threads(int work_size) {
  int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return std::max(std::min(1 << pow_2, kTotalThreads), 1);
}

template <unsigned int block_size, typename scalar_t>
__global__ void furthest_point_sampling_cuda_kernel(
    int64_t batch_size, int64_t num_points, int64_t num_samples,
    int64_t* __restrict__ outputs_ptr,
    scalar_t* __restrict__ coords_ptr,
    scalar_t* __restrict__ tmp_buf_ptr) {
  if (num_samples <= 0) {
    return;
  }

  __shared__ float dists[block_size];
  __shared__ int dists_idx[block_size];

  int batch_index = blockIdx.x;
  coords_ptr += batch_index * num_points * 3;
  tmp_buf_ptr += batch_index * num_points;
  outputs_ptr += batch_index * num_samples;

  const int tid = threadIdx.x;
  const int stride = block_size;

  int idx = 0;
  if (threadIdx.x == 0) outputs_ptr[0] = 0;

  __syncthreads();

  for (int j = 1; j < num_samples; j++) {
    int best_idx = 0;
    float best_val = -1;

    const float x1 = coords_ptr[idx * 3 + 0];
    const float y1 = coords_ptr[idx * 3 + 1];
    const float z1 = coords_ptr[idx * 3 + 2];

    for (int k = tid; k < num_points; k += stride) {
      const float x2 = coords_ptr[k * 3 + 0];
      const float y2 = coords_ptr[k * 3 + 1];
      const float z2 = coords_ptr[k * 3 + 2];

      auto mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (mag < 1e-3) {
        continue;
      }

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      d = min(d, tmp_buf_ptr[k]);
      tmp_buf_ptr[k] = d;

      if (d > best_val) {
        best_idx = k;
        best_val = d;
      }
    }
    dists[tid] = best_val;
    dists_idx[tid] = best_idx;
    __syncthreads();

    auto update = [] (float *__restrict__ dists, int *__restrict__ dists_idx, int idx1, int idx2) {
      const float v1 = dists[idx1], v2 = dists[idx2];
      const int i1 = dists_idx[idx1], i2 = dists_idx[idx2];

      if (v2 > v1) {
        dists[idx1] = v2;
        dists_idx[idx1] = i2;
      }
    };

    if (block_size >= 512) {
      if (tid < 256) {
        update(dists, dists_idx, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        update(dists, dists_idx, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        update(dists, dists_idx, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        update(dists, dists_idx, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        update(dists, dists_idx, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        update(dists, dists_idx, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        update(dists, dists_idx, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        update(dists, dists_idx, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        update(dists, dists_idx, tid, tid + 1);
      }
      __syncthreads();
    }

    idx = dists_idx[0];
    if (tid == 0) {
      outputs_ptr[j] = idx;
    }
  }
}

template <typename scalar_t>
void furthest_point_sampling_cuda_impl(
    at::Tensor& outputs,
    const at::Tensor& coords,
    int64_t num_samples) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto batch_size = coords.size(0);
  auto num_points = coords.size(1);

  auto tmp_buf = at::full({batch_size, num_points}, 1e10f, coords.options());

  auto outputs_ptr = outputs.data_ptr<int64_t>();
  auto coords_ptr = coords.data_ptr<scalar_t>();
  auto tmp_buf_ptr = tmp_buf.data_ptr<scalar_t>();

  auto num_threads = opt_num_threads(num_points);

  switch (num_threads) {
    case 512:
      furthest_point_sampling_cuda_kernel<512>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 256:
      furthest_point_sampling_cuda_kernel<256>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 128:
      furthest_point_sampling_cuda_kernel<128>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 64:
      furthest_point_sampling_cuda_kernel<64>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 32:
      furthest_point_sampling_cuda_kernel<32>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 16:
      furthest_point_sampling_cuda_kernel<16>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 8:
      furthest_point_sampling_cuda_kernel<8>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 4:
      furthest_point_sampling_cuda_kernel<4>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 2:
      furthest_point_sampling_cuda_kernel<2>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    case 1:
      furthest_point_sampling_cuda_kernel<1>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
      break;
    default:
      furthest_point_sampling_cuda_kernel<512>
          <<<batch_size, num_threads, 0, stream>>>(
            batch_size, num_points, num_samples, outputs_ptr, coords_ptr, tmp_buf_ptr);
  }

  CUDA_CHECK_ERRORS();
}

at::Tensor furthest_point_sampling_cuda(
    const at::Tensor& coords, int64_t num_samples) {
  TORCH_CHECK(coords.is_cuda(), "coords must be a CUDA tensor");

  auto outputs = at::empty(
    {coords.size(0), num_samples}, coords.options().dtype(at::ScalarType::Long));

  AT_DISPATCH_FLOATING_TYPES(coords.type(), "furthest_point_sampling_cuda", [&] {
    furthest_point_sampling_cuda_impl<scalar_t>(outputs, coords, num_samples);
  });

  return outputs;
}

TORCH_LIBRARY_IMPL(diffusers_3d, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("diffusers_3d::furthest_point_sampling"),
         TORCH_FN(furthest_point_sampling_cuda));
}

} // namespace diffusers_3d::sampling
