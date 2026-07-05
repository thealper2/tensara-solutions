#include <cuda_runtime.h>
#include <math.h>

__global__ void mse_sum_kernel(const float* __restrict__ predictions,
                               const float* __restrict__ targets,
                               float* __restrict__ sum_sq, size_t total_elements) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    
    while (idx < total_elements) {
        float diff = predictions[idx] - targets[idx];
        sum += diff * diff;
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

extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim) {
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; d++) {
        total_elements *= shape[d];
    }
    
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    mse_sum_kernel<<<grid_size, block_size>>>(predictions, targets, d_sum, total_elements);
    cudaDeviceSynchronize();
    
    float h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    float mse = h_sum / total_elements;
    cudaMemcpy(output, &mse, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaFree(d_sum);
}
