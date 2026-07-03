#include <cuda_runtime.h>

__global__ void hard_sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m) {
    int total_elements = n * m;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float val = input[idx] / 6.0f + 0.5f;
    output[idx] = fminf(fmaxf(val, 0.0f), 1.0f);
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    hard_sigmoid_kernel<<<grid_size, block_size>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
