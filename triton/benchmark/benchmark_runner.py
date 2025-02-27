import torch
import triton
import triton.language as tl


@triton.jit
def square_matrix_kernel(matrix_ptr, result_ptr, width: tl.constexpr, height: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < height)[:, None] & (col < width)[None, :]

    idx = row[:, None] * width + col[None, :]

    matrix = tl.load(matrix_ptr + idx, mask=mask)
    tl.store(result_ptr + idx, matrix * matrix, mask=mask)


def square_matrix_triton(matrix):
    height, width = matrix.shape
    result = torch.empty_like(matrix)
    BLOCK_SIZE = 16

    grid = (height // BLOCK_SIZE + 1, width // BLOCK_SIZE + 1)
    square_matrix_kernel[grid](
        matrix, result, width, height, BLOCK_SIZE=BLOCK_SIZE
    )

    return result

def benchmark_kernel(op, input_tensor, num_iters=100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(10):  # Warm-up
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

def pytorch_op(input_tensor):
    return input_tensor ** 2

# Benchmark Triton kernel vs PyTorch built-in operation
triton_time = benchmark_kernel(square_matrix_triton, input_tensor)
pytorch_time = benchmark_kernel(pytorch_op, input_tensor)

print(f"Triton Kernel Time: {triton_time:.4f} ms")
print(f"PyTorch Built-in Time: {pytorch_time:.4f} ms")
print(f"Speedup: {pytorch_time / triton_time:.2f}x")
