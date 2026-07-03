#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * m;

    if (idx < total_elements) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size>>>(input, output, (int)n, int(m));
    cudaDeviceSynchronize();
}
