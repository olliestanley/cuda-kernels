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
#define TILE_A_ROWS 128
#define TILE_A_COLS 16

// cols per tile of matrix B
#define TILE_B_COLS 128

__global__ void matmul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols) {
    const int n_threads_per_block = TILE_A_ROWS * TILE_B_COLS / (ROWS_PER_THREAD * COLS_PER_THREAD);
    static_assert(n_threads_per_block % TILE_A_COLS == 0);
    static_assert(n_threads_per_block % TILE_B_COLS == 0);

    // per-thread view indices for loading A tiles
    const int A_view_ty = threadIdx.x / TILE_A_COLS;
    const int A_view_tx = threadIdx.x % TILE_A_COLS;
    const int A_stride = n_threads_per_block / TILE_A_COLS;

    // per-thread view indices for loading B tiles
    const int B_view_ty = threadIdx.x / TILE_B_COLS;
    const int B_view_tx = threadIdx.x % TILE_B_COLS;
    const int B_stride = n_threads_per_block / TILE_B_COLS;

    // starting indices for the registers that accumulate the dot product
    const int row_start = COLS_PER_THREAD * (threadIdx.x / (TILE_B_COLS / ROWS_PER_THREAD));
    const int col_start = ROWS_PER_THREAD * (threadIdx.x % (TILE_B_COLS / ROWS_PER_THREAD));

    // repeated memory accesses to global memory are slow -> use shared memory for tiles
    // shared memory is small -> use tiles rather than the whole matrix
    __shared__ float A_shared[TILE_A_ROWS][TILE_A_COLS];
    __shared__ float B_shared[TILE_A_COLS][TILE_B_COLS];

    const int num_tiles = ceil(A_n_cols / (float) TILE_A_COLS);

    // accumulate COLS_PER_THREADxROWS_PER_THREAD values in registers before writing to global memory
    float value[COLS_PER_THREAD][ROWS_PER_THREAD] = {0.0f};

    float A_temp[ROWS_PER_THREAD] = {0.0f};
    float B_temp[COLS_PER_THREAD] = {0.0f};

    for (int tile = 0; tile < num_tiles; tile++) {
        // load tiles into shared memory first
        for (int offset = 0; offset < TILE_A_ROWS; offset += A_stride) {
            int global_row = blockIdx.y * TILE_A_ROWS + offset + A_view_ty;
            int global_col = tile * TILE_A_COLS + A_view_tx;
            if (global_row < C_n_rows && global_col < A_n_cols) {
                A_shared[offset + A_view_ty][A_view_tx] = d_A_ptr[global_row * A_n_cols + global_col];
            } else {
                A_shared[offset + A_view_ty][A_view_tx] = 0.0f;
            }
        }
        for (int offset = 0; offset < TILE_A_COLS; offset += B_stride) {
            int global_row = tile * TILE_A_COLS + B_view_ty + offset;
            int global_col = blockIdx.x * TILE_B_COLS + B_view_tx;
            if (global_row < A_n_cols && global_col < C_n_cols) {
                B_shared[B_view_ty + offset][B_view_tx] = d_B_ptr[global_row * C_n_cols + global_col];
            } else {
                B_shared[B_view_ty + offset][B_view_tx] = 0.0f;
            }
        }
        // sync ensures all threads have loaded tiles into shared memory
        __syncthreads();

        for (int j = 0; j < TILE_A_COLS; j++) {
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
            int global_row = blockIdx.y * TILE_A_ROWS + row_start + y;
            int global_col = blockIdx.x * TILE_B_COLS + col_start + x;
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

    dim3 dim_block(TILE_A_ROWS * TILE_B_COLS / (ROWS_PER_THREAD * COLS_PER_THREAD));
    dim3 dim_grid(ceil(C_n_cols / (float)(TILE_B_COLS)), ceil(C_n_rows / (float)(TILE_A_ROWS)));
    matmul_kernel<<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols);
}
