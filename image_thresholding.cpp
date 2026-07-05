#include <cuda_runtime.h>

__global__ void image_threshold_kernel(const float* __restrict__ input_image, float threshold_value, float* __restrict__ output_image, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < W && y < H) {
        int index = y * W + x;

        if (input_image[index] > threshold_value) {
            output_image[index] = 255.0f;
        } else {
            output_image[index] = 0.0f;
        }
    }
}

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    int H = (int)height;
    int W = (int)width;

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    image_threshold_kernel<<<grid, block>>>(input_image, threshold_value, output_image, H, W);
    cudaDeviceSynchronize();
}
