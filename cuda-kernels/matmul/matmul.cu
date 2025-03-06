// TODO: further optimise using more registers & vectorized memory accesses

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define ELEMENTS_PER_THREAD 8
// Tiles of matrix A
#define tiles_A_rows 64
#define tiles_A_cols 8
// Tiles of matrix B
#define tiles_B_cols 64

__global__ void matmul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols) {
    const int A_view_ty = threadIdx.x / tiles_A_cols;
    const int A_view_tx = threadIdx.x % tiles_A_cols;
    const int B_view_ty = threadIdx.x / tiles_B_cols;
    const int B_view_tx = threadIdx.x % tiles_B_cols;

    const int row_start = tiles_A_rows * blockIdx.y + ELEMENTS_PER_THREAD * (threadIdx.x / tiles_B_cols);
    const int col = tiles_B_cols * blockIdx.x + (threadIdx.x % tiles_B_cols);

    // repeated memory accesses to global memory are slow -> use shared memory
    // shared memory is small -> use tiles rather than the whole matrix
    __shared__ float A_shared[tiles_A_rows][tiles_A_cols];
    __shared__ float B_shared[tiles_A_cols][tiles_B_cols];

    const int num_tiles = ceil(A_n_cols / (float) tiles_A_cols);

    float value[ELEMENTS_PER_THREAD] = {0.0f};
    for (int tile = 0; tile < num_tiles; tile++) {
        // load tiles into shared memory
        int aRow = blockIdx.y * tiles_A_rows + A_view_ty;
        int aCol = tile * tiles_A_cols + A_view_tx;
        if (aRow < C_n_rows && aCol < A_n_cols) {
            A_shared[A_view_ty][A_view_tx] = d_A_ptr[aRow * A_n_cols + aCol];
        } else {
            A_shared[A_view_ty][A_view_tx] = 0.0f;
        }
        int bRow = tile * tiles_A_cols + B_view_ty;
        int bCol = blockIdx.x * tiles_B_cols + B_view_tx;
        if (bRow < A_n_cols && bCol < C_n_cols) {
            B_shared[B_view_ty][B_view_tx] = d_B_ptr[bRow * C_n_cols + bCol];
        } else {
            B_shared[B_view_ty][B_view_tx] = 0.0f;
        }
        // sync ensures all threads have loaded tiles into shared memory
        __syncthreads();

        for (int j = 0; j < tiles_A_cols; j++) {
            float B_val_register = B_shared[j][B_view_tx];
            // compute dot product
            for (int c = 0; c < ELEMENTS_PER_THREAD; c++) {
                value[c] += A_shared[B_view_ty * ELEMENTS_PER_THREAD + c][j] * B_val_register;
            }
        }
        // sync ensures all threads have finished computation
        __syncthreads();
    }

    for (int c = 0; c < ELEMENTS_PER_THREAD; c++) {
        if (row_start + c < C_n_rows && col < C_n_cols) {
            d_C_ptr[(row_start + c) * C_n_cols + col] = 1 * value[c] + 0 * d_C_ptr[(row_start + c) * C_n_cols + col];
        }
    }
}

void matmul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    int C_n_rows = a.size(0);
    int C_n_cols = b.size(1);
    int A_n_cols = a.size(1);

    dim3 dim_block(tiles_A_rows * tiles_B_cols / ELEMENTS_PER_THREAD);
    dim3 dim_grid(ceil(C_n_cols / (float)(tiles_B_cols)), ceil(C_n_rows / (float)(tiles_A_rows)));
    matmul_kernel<<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols);
}
