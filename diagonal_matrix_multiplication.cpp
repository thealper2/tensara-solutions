#include <cuda_runtime.h>

__global__ void diagonal_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= M) return;

    C[row * M + col] = A[row] * B[row * M + col];
}

extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m) {
    int N = (int)n;
    int M = (int)m;

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    diagonal_matmul_kernel<<<grid, block>>>(diagonal_a, input_b, output_c, N, M);
    cudaDeviceSynchronize();
}
