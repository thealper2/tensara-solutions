#include <cuda_runtime.h>

__global__ void sigmoid_kernel(const float* input, float* output, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * m;

    if (idx < total_elements) {
        float val = -input[idx];
        output[idx] = 1.0f / (1.0f + expf(val));
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    sigmoid_kernel<<<grid_size, block_size>>>(input, output, (int)n, (int)m);
    cudaDeviceSynchronize();
}
