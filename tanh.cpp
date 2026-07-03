#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* __restrict__ input, float* __restrict__ output, int n, int m) {
    int total_elements = n * m;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float val = input[idx];
        output[idx] = tanhf(val);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    tanh_kernel<<<grid_size, block_size>>>(input, output, (int)n, (int)m);
    cudaDeviceSynchronize();
}
