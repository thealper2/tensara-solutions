#include <cuda_runtime.h>

__global__ void elu_kernel(const float* __restrict__ input, float* __restrict__ output, int n, int m, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * m;

    if (idx < total_elements) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val: (alpha * (expf(val) - 1.0f));
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    elu_kernel<<<grid_size, block_size>>>(input, output, (int)n, (int)m, alpha);
    cudaDeviceSynchronize();
}
