#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, int n, int m) {
    int total_elements = n * m;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    gelu_kernel<<<grid_size, block_size>>>(input, output, (int)n, (int)m);
    cudaDeviceSynchronize();
}
