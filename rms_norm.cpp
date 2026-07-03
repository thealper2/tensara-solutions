#include <cuda_runtime.h>

__global__ void rms_norm_kernel(const float* X, float* Y, int N, float eps) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    __shared__ float shared_sum[256];
    float sum = 0.0f;
    
    for (int i = thread_idx; i < N; i += blockDim.x) {
        float val = X[batch_idx * N + i];
        sum += val * val;
    }
    
    shared_sum[thread_idx] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_sum[thread_idx] += shared_sum[thread_idx + s];
        }
        __syncthreads();
    }
    
    float total_sum = shared_sum[0];
    float rms = sqrtf(total_sum / N + eps);
    float inv_rms = 1.0f / rms;
    
    for (int i = thread_idx; i < N; i += blockDim.x) {
        Y[batch_idx * N + i] = X[batch_idx * N + i] * inv_rms;
    }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    int block_size = 256;
    int grid_size = (int)B;
    float eps = 1e-5f;
    
    rms_norm_kernel<<<grid_size, block_size>>>(X, Y, (int)N, eps);
    cudaDeviceSynchronize();
}
