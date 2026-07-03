#include <cuda_runtime.h>

__global__ void sum_reduce_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  size_t outer, size_t reduce, size_t inner) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * inner;
    if (idx >= total) return;

    size_t o = idx / inner;
    size_t i = idx % inner;

    const float* base = input + o * reduce * inner + i;
    float sum = 0.0f;
    for (size_t r = 0; r < reduce; r++) {
        sum += base[r * inner];
    }
    output[o * inner + i] = sum;
}

extern "C" void solution(const float* input, int dim, float* output, const size_t* shape, size_t ndim) {
    size_t h_shape[16];
    cudaMemcpy(h_shape, shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t outer = 1, reduce = h_shape[dim], inner = 1;
    for (int i = 0; i < dim; i++) outer *= h_shape[i];
    for (size_t i = dim + 1; i < ndim; i++) inner *= h_shape[i];

    size_t total = outer * inner;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    sum_reduce_kernel<<<blocks, threads>>>(input, output, outer, reduce, inner);
}
