#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* __restrict__ input, const float* __restrict__ mask, float* __restrict__ output, int input_len, int mask_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < input_len) {
        float sum = 0.0f;
        int radius = mask_len / 2;
        
        for (int j = 0; j < mask_len; j++) {
            int input_idx = i + j - radius;
            
            if (input_idx >= 0 && input_idx < input_len) {
                sum += input[input_idx] * mask[j];
            }
        }
        output[i] = sum;
    }
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    conv1d_kernel<<<grid_size, block_size>>>(A, B, C, (int)N, (int)K);
    cudaDeviceSynchronize();
}
