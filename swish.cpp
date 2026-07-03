#include <cuda_runtime.h>

__global__ void swish_kernel(const float* __restrict__ input, float* __restrict__ output, int n, int m) {
    int total_elements = n * m;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_elements) return;
    float x = input[idx];
    output[idx] = x / (1.0f + expf(-x));
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    swish_kernel<<<grid_size, block_size>>>(input, output, (int)n, (int)m);
    cudaDeviceSynchronize();
}
