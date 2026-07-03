#include <cuda_runtime.h>

__global__ void softplus_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m) {
    int total_elements = n * m;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = input[idx];
    if (val > 0.0f) {
        output[idx] = val + logf(1.0f + expf(-val));
    } else {
        output[idx] = logf(1.0f + expf(val));
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    softplus_kernel<<<grid_size, block_size>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
