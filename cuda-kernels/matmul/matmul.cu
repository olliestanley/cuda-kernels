// TODO: further optimise using vectorized memory accesses

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// first implementation just used 8 elements per thread
// to utilise registers more heavily, we use 8x8
#define ROWS_PER_THREAD 8
#define COLS_PER_THREAD 8

// rows & cols per tile of matrix A
#define tiles_A_rows 128
#define tiles_A_cols 16

// cols per tile of matrix B
#define tiles_B_cols 128

__global__ void matmul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols) {
    const int n_threads_per_block = tiles_A_rows * tiles_B_cols / (ROWS_PER_THREAD * COLS_PER_THREAD);
    static_assert(n_threads_per_block % tiles_A_cols == 0);
    static_assert(n_threads_per_block % tiles_B_cols == 0);

    // per-thread view indices for loading A tiles
    const int A_view_ty = threadIdx.x / tiles_A_cols;
    const int A_view_tx = threadIdx.x % tiles_A_cols;
    const int A_stride = n_threads_per_block / tiles_A_cols;

    // per-thread view indices for loading B tiles
    const int B_view_ty = threadIdx.x / tiles_B_cols;
    const int B_view_tx = threadIdx.x % tiles_B_cols;
    const int B_stride = n_threads_per_block / tiles_B_cols;

    // starting indices for the registers that accumulate the dot product
    const int row_start = COLS_PER_THREAD * (threadIdx.x / (tiles_B_cols / ROWS_PER_THREAD));
    const int col_start = ROWS_PER_THREAD * (threadIdx.x % (tiles_B_cols / ROWS_PER_THREAD));

    // repeated memory accesses to global memory are slow -> use shared memory for tiles
    // shared memory is small -> use tiles rather than the whole matrix
    __shared__ float A_shared[tiles_A_rows][tiles_A_cols];
    __shared__ float B_shared[tiles_A_cols][tiles_B_cols];

    const int num_tiles = ceil(A_n_cols / (float) tiles_A_cols);

    // accumulate COLS_PER_THREADxROWS_PER_THREAD values in registers before writing to global memory
    float value[COLS_PER_THREAD][ROWS_PER_THREAD] = {0.0f};

    float A_temp[ROWS_PER_THREAD] = {0.0f};
    float B_temp[COLS_PER_THREAD] = {0.0f};

    for (int tile = 0; tile < num_tiles; tile++) {
        // load tiles into shared memory first
        for (int offset = 0; offset < tiles_A_rows; offset += A_stride) {
            int global_row = blockIdx.y * tiles_A_rows + offset + A_view_ty;
            int global_col = tile * tiles_A_cols + A_view_tx;
            if (global_row < C_n_rows && global_col < A_n_cols) {
                A_shared[offset + A_view_ty][A_view_tx] = d_A_ptr[global_row * A_n_cols + global_col];
            } else {
                A_shared[offset + A_view_ty][A_view_tx] = 0.0f;
            }
        }
        for (int offset = 0; offset < tiles_A_cols; offset += B_stride) {
            int global_row = tile * tiles_A_cols + B_view_ty + offset;
            int global_col = blockIdx.x * tiles_B_cols + B_view_tx;
            if (global_row < A_n_cols && global_col < C_n_cols) {
                B_shared[B_view_ty + offset][B_view_tx] = d_B_ptr[global_row * C_n_cols + global_col];
            } else {
                B_shared[B_view_ty + offset][B_view_tx] = 0.0f;
            }
        }
        // sync ensures all threads have loaded tiles into shared memory
        __syncthreads();

        for (int j = 0; j < tiles_A_cols; j++) {
            // load the block into registers, avoiding repeated shared memory accesses
            for (int i = 0; i < ROWS_PER_THREAD; i++) {
                A_temp[i] = A_shared[row_start + i][j];
            }
            for (int i = 0; i < COLS_PER_THREAD; i++) {
                B_temp[i] = B_shared[j][col_start + i];
            }

            // compute dot product
            for (int y = 0; y < COLS_PER_THREAD; y++) {
                for (int x = 0; x < ROWS_PER_THREAD; x++) {
                    value[y][x] += A_temp[y] * B_temp[x];
                }
            }
        }
        // sync ensures all threads have finished computation
        __syncthreads();
    }

    // finally, write to global memory
    for (int y = 0; y < COLS_PER_THREAD; y++) {
        for (int x = 0; x < ROWS_PER_THREAD; x++) {
            int global_row = blockIdx.y * tiles_A_rows + row_start + y;
            int global_col = blockIdx.x * tiles_B_cols + col_start + x;
            if (global_row < C_n_rows && global_col < C_n_cols) {
                d_C_ptr[global_row * C_n_cols + global_col] = 1 * value[y][x] + 0 * d_C_ptr[global_row * C_n_cols + global_col];
            }
        }
    }
}

void matmul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    int C_n_rows = a.size(0);
    int C_n_cols = b.size(1);
    int A_n_cols = a.size(1);

    dim3 dim_block(tiles_A_rows * tiles_B_cols / (ROWS_PER_THREAD * COLS_PER_THREAD));
    dim3 dim_grid(ceil(C_n_cols / (float)(tiles_B_cols)), ceil(C_n_rows / (float)(tiles_A_rows)));
    matmul_kernel<<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols);
}
