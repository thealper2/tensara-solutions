#include <cuda_runtime.h>

__global__ void log_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, size_t M, size_t N) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= M) return;

    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    float max_val = -INFINITY;
    for (int i = col; i < N; i += blockDim.x) {
        float val = input[row * N + i];
        if (val > max_val) max_val = val;
    }
    shared_max[col] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (col < s) {
            if (shared_max[col + s] > shared_max[col]) {
                shared_max[col] = shared_max[col + s];
            }
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = col; i < N; i += blockDim.x) {
        sum += expf(input[row * N + i] - max_val);
    }
    shared_sum[col] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (col < s) {
            shared_sum[col] += shared_sum[col + s];
        }
        __syncthreads();
    }
    sum = shared_sum[0];
    __syncthreads();
    
    float log_sum = logf(sum);
    for (int i = col; i < N; i += blockDim.x) {
        output[row * N + i] = input[row * N + i] - max_val - log_sum;
    }
}

extern "C" void solution(const float* input, float* output, size_t M, size_t N) {
    int block_size = 256;
    int grid_size = M;

    log_softmax_kernel<<<grid_size, block_size>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
