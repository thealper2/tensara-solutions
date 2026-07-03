#include <cuda_runtime.h>

__global__ void huber_loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float diff = predictions[idx] - targets[idx];
    float abs_diff = fabsf(diff);

    if (abs_diff < 1.0f) {
        output[idx] = 0.5f * diff * diff;
    } else {
        output[idx] = abs_diff - 0.5f;
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    huber_loss_kernel<<<grid_size, block_size>>>(predictions, targets, output, n);
    cudaDeviceSynchronize();
}
