#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void grayscale_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;
    const int tid = threadIdx.x;

    #pragma unroll 4
    for (int i = (block_idx * block_size + tid) * 4; i < N; i += grid_size * block_size * 4) {
        // read 12 element in 3 loads
        const float4 val1 = reinterpret_cast<const float4 *>(input + i * 3)[0];
        const float4 val2 = reinterpret_cast<const float4 *>(input + i * 3)[1];
        const float4 val3 = reinterpret_cast<const float4 *>(input + i * 3)[2];

        float4 out;
        out.x = fmaf(0.2989f, val1.x, fmaf(0.5870f, val1.y, 0.1140f * val1.z));
        out.y = fmaf(0.2989f, val1.w, fmaf(0.5870f, val2.x, 0.1140f * val2.y));
        out.z = fmaf(0.2989f, val2.z, fmaf(0.5870f, val2.w, 0.1140f * val3.x));
        out.w = fmaf(0.2989f, val3.y, fmaf(0.5870f, val3.z, 0.1140f * val3.w));

        reinterpret_cast<float4 *>(output + i)[0] = out;
    }
}

void grayscale(torch::Tensor input, torch::Tensor output) {
    const int N = output.numel();
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, grayscale_kernel);
    grayscale_kernel<<<minGridSize, blockSize>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
}
