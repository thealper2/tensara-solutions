#include <cuda_runtime.h>

__global__ void cosine_similarity_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float dot = 0.0f;
    float norm_pred = 0.0f;
    float norm_target = 0.0f;

    int base = idx * d;
    for (int i = 0; i < d; i++) {
        float p = predictions[base + i];
        float t = targets[base + i];
        dot += p * t;
        norm_pred += p * p;
        norm_target += t * t;
    }

    norm_pred = sqrtf(norm_pred);
    norm_target = sqrtf(norm_target);

    float eps = 1e-8f;
    norm_pred = norm_pred > eps ? norm_pred : eps;
    norm_target = norm_target > eps ? norm_target : eps;

    float similarity = dot / (norm_pred * norm_target);
    output[idx] = 1.0f - similarity;
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    cosine_similarity_kernel<<<grid_size, block_size>>>(predictions, targets, output, (int)n, (int)d);
    cudaDeviceSynchronize();
}
