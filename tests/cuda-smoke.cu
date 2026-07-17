#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

__global__ void saxpy(const float* x, float* y, float alpha, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        y[index] = alpha * x[index] + y[index];
    }
}

int main() {
    constexpr int count = 4096;
    constexpr float alpha = 2.0F;
    constexpr size_t bytes = count * sizeof(float);
    float *x = nullptr, *y = nullptr;

    if (cudaMallocManaged(&x, bytes) != cudaSuccess ||
        cudaMallocManaged(&y, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged failed\n");
        return 1;
    }

    for (int i = 0; i < count; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = 1.0F;
    }

    saxpy<<<(count + 255) / 256, 256>>>(x, y, alpha, count);
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(status));
        return 1;
    }

    const float expected = alpha * static_cast<float>(count - 1) + 1.0F;
    const bool valid = std::fabs(y[count - 1] - expected) < 0.001F;
    cudaFree(x);
    cudaFree(y);
    std::printf("nvcc CUDA SAXPY: %s\n", valid ? "PASS" : "FAIL");
    return valid ? 0 : 1;
}
