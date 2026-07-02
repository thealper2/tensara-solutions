#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    vector_add_kernel<<<grid_size, block_size>>>(d_input1, d_input2, d_output, n);
    cudaDeviceSynchronize();
}
