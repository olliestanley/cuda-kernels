import torch
from torch.utils import cpp_extension

cpp_src = """
torch::Tensor square_matrix(torch::Tensor matrix);
"""

cuda_src = """
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
}
"""

def load_inline(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return cpp_extension.load_inline(
        name="custom_kernel",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=funcs,
        extra_cuda_cflags=["--O2"] if opt else [],
        verbose=verbose,
    )

custom_extension = load_inline(
    cuda_src=cuda_src,
    cpp_src=cpp_src,
    funcs=["square_matrix"],
)

def benchmark_kernel(op, input_tensor, num_iters=100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        op(input_tensor)

    torch.cuda.synchronize()

    start_event.record()
    for _ in range(num_iters):
        op(input_tensor)
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters

size = (1 << 8, 1 << 8)
input_tensor = torch.ones(size, dtype=torch.float32, device="cuda")

def custom_op(input_tensor):
    return custom_extension.square_matrix(input_tensor)

def pytorch_op(input_tensor):
    return input_tensor ** 2

# Benchmark custom kernel vs PyTorch build-in operation
custom_time = benchmark_kernel(custom_op, input_tensor)
pytorch_time = benchmark_kernel(pytorch_op, input_tensor)

print(f"Custom CUDA Kernel Time: {custom_time:.4f} ms")
print(f"PyTorch Built-in Time: {pytorch_time:.4f} ms")
print(f"Speedup: {pytorch_time / custom_time:.2f}x")
