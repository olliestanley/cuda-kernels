from pathlib import Path

import torch
from torch.utils import cpp_extension

current_directory = Path(__file__).resolve().parent
cpp_src = (current_directory / "square_matrix.cpp").read_text()
cuda_src = (current_directory / "square_matrix.cu").read_text()

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
