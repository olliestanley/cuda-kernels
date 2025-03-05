// TODO: further optimise using registers and vectorized memory accesses

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// tile width for shared memory usage, as we cannot fit the whole matrix into shared memory
#define TILE_WIDTH 32

__global__ void matmul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols) {
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);

    // using x for col and y for row helps with coalescing (how efficiently threads access global memory)
    // this way we access two of three matrices in a row-major order, but still not all three
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    // repeated memory accesses to global memory are slow -> use shared memory
    // shared memory is small -> use tiles rather than the whole matrix
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    const int num_tiles = ceil(A_n_cols / (float)TILE_WIDTH);

    float value = 0.0f;
    for (int i = 0; i < num_tiles; i++) {
        // load tiles into shared memory
        if (row < C_n_rows && i * TILE_WIDTH + threadIdx.x < A_n_cols) {
            A_shared[threadIdx.y][threadIdx.x] = d_A_ptr[row * A_n_cols + i * TILE_WIDTH + threadIdx.x];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < C_n_cols && i * TILE_WIDTH + threadIdx.y < A_n_cols) {
            B_shared[threadIdx.y][threadIdx.x] = d_B_ptr[(i * TILE_WIDTH + threadIdx.y) * C_n_cols + col];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // sync ensures all threads have loaded tiles into shared memory
        __syncthreads();

        // compute dot product of tiles
        for (int j = 0; j < TILE_WIDTH; j++) {
            if (row < C_n_rows && col < C_n_cols) {
                value += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
            }
        }
        // sync ensures all threads have finished computation
        __syncthreads();
    }

    if (row < C_n_rows && col < C_n_cols) {
        d_C_ptr[row * C_n_cols + col] = 1 * value + 0 * d_C_ptr[row * C_n_cols + col];
    }
}

void matmul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    int C_n_rows = a.size(0);
    int C_n_cols = b.size(1);
    int A_n_cols = a.size(1);

    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(C_n_rows / (float)(32)), ceil(C_n_cols / (float)(32)), 1);
    matmul_kernel<<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols);
}
