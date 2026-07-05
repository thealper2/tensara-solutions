#include <cuda_runtime.h>
#include <math.h>

__global__ void frobenius_norm_kernel(const float* __restrict__ X, float* __restrict__ Y, float* __restrict__ sum_sq, size_t size) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    while (idx < size) {
        float val = X[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_sq, shared_sum[0]);
    }
}

__global__ void normalize_kernel(const float* __restrict__ X, float* __restrict__ Y, float norm, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    Y[idx] = X[idx] / norm;
}

extern "C" void solution(const float* X, float* Y, size_t size) {
    float* d_sum_sq;
    cudaMalloc(&d_sum_sq, sizeof(float));
    cudaMemset(d_sum_sq, 0, sizeof(float));

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    frobenius_norm_kernel<<<grid_size, block_size>>>(X, Y, d_sum_sq, size);
    cudaDeviceSynchronize();

    float h_sum_sq;
    cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);

    float norm = sqrtf(h_sum_sq);
    if (norm < 1e-8f) norm = 1.0f;

    normalize_kernel<<<grid_size, block_size>>>(X, Y, norm, size);
    cudaDeviceSynchronize();

    cudaFree(d_sum_sq);
}
