#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");
    auto c = torch::empty_like(a);
    int n = a.numel();
    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    cudaDeviceSynchronize();
    return c;
}
