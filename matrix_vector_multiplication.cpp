#include <cuda_runtime.h>

__global__ void matrix_vector_mult(const float *d_matrix, const float* d_vector, float* d_result, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += d_matrix[row * N + col] * d_vector[col];
        }
        d_result[row] = sum;
    }
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    int total_elements = m * k;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    matrix_vector_mult<<<grid_size, block_size>>>(input_a, input_b, output_c, m, k);
    cudaDeviceSynchronize();
}
