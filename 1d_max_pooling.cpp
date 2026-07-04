#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_1d_kernel(const float* __restrict__ input, int kernel_size, int stride, int padding, int dilation, float* __restrict__ output, size_t H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = ((int)H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    if (i >= output_size) return;

    float max_val = -FLT_MAX;
    int start = i * stride - padding;
    int effective_kernel = (kernel_size - 1) * dilation + 1;

    for (int j = 0; j < kernel_size; j++) {
        int idx = start + j * dilation;
        if (idx >= 0 && idx < (int)H) {
            float val = input[idx];
            if (val > max_val) max_val = val;
        }
    }

    output[i] = max_val;
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
    int output_size = ((int)H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;

    max_pool_1d_kernel<<<grid_size, block_size>>>(input, kernel_size, stride, padding, dilation, output, H);
    cudaDeviceSynchronize();
}
