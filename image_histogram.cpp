#include <cuda_runtime.h>

__global__ void histogram_kernel(const float* __restrict__ image, int num_bins, float* __restrict__ histogram, size_t height, size_t width) {
    __shared__ float shared_hist[256];
    int tid = threadIdx.x;

    if (tid < num_bins) {
        shared_hist[tid] = 0.0f;
    }
    __syncthreads();

    size_t total_pixels = height * width;
    size_t idx = blockIdx.x * blockDim.x + tid;

    while (idx < total_pixels) {
        int pixel_val = (int)image[idx];
        if (pixel_val >= 0 && pixel_val < num_bins) {
            atomicAdd(&shared_hist[pixel_val], 1.0f);
        }
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();

    if (tid < num_bins) {
        atomicAdd(&histogram[tid], shared_hist[tid]);
    }
}

extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) {
    cudaMemset(histogram, 0, num_bins * sizeof(float));

    size_t total_pixels = height * width;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    histogram_kernel<<<grid_size, block_size>>>(image, num_bins, histogram, height, width);
    cudaDeviceSynchronize();
}
