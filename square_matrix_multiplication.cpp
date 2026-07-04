#include <cuda_runtime.h>

__global__ void matmul_square_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    int N = (int)n;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_square_kernel<<<grid, block>>>(input_a, input_b, output_c, N);
    cudaDeviceSynchronize();
}
