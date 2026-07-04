#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = predictions[idx];
    float y = targets[idx];
    float margin = 1.0f - y * x;

    output[idx] = margin > 0.0f ? margin : 0.0f;
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hinge_loss_kernel<<<grid_size, block_size>>>(predictions, targets, output, (int)n);
    cudaDeviceSynchronize();
}
