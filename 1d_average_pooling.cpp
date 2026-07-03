#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(const float* input, float* output, int H, int kernel_size, int stride, int padding, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        float sum = 0.0f;
        int start = i * stride - padding;
        
        for (int j = 0; j < kernel_size; j++) {
            int idx = start + j;
            if (idx >= 0 && idx < H) {
                sum += input[idx];
            }
        }
        output[i] = sum / kernel_size;
    }
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    int output_size = (H + 2 * padding - kernel_size) / stride + 1;
    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;

    avg_pool1d_kernel<<<grid_size, block_size>>>(input, output, (int)H, kernel_size, stride, padding, output_size);
    cudaDeviceSynchronize();
}
