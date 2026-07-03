#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float alpha, float* output, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * m;

    if (idx < total_elements) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : (val * alpha); 
    }
}

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    leaky_relu_kernel<<<grid_size, block_size>>>(input, alpha, output, (int)n, (int)m);
    cudaDeviceSynchronize();
}
