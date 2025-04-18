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

template <bool AlphaOne, bool BetaZero>

__global__ void matmul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols, float alpha, float beta) {
    const int n_threads_per_block = TILE_A_ROWS * TILE_B_COLS / (ROWS_PER_THREAD * COLS_PER_THREAD);
    static_assert(n_threads_per_block % TILE_A_COLS == 0);
    static_assert(n_threads_per_block % TILE_B_COLS == 0);

    // we will use vectorised memory accesses of 4, hence these checks
    static_assert(TILE_A_COLS % 4 == 0);
    static_assert(TILE_B_COLS % 4 == 0);
    assert(C_n_rows % 4 == 0);
    assert(C_n_cols % 4 == 0);
    assert(A_n_cols % 4 == 0);

    // per-thread view indices for loading A tiles
    const int A_view_ty = threadIdx.x / (TILE_A_COLS / 4);
    const int A_view_tx = threadIdx.x % (TILE_A_COLS / 4);
    const int A_stride = n_threads_per_block / (TILE_A_COLS / 4);

    // per-thread view indices for loading B tiles
    const int B_view_ty = threadIdx.x / (TILE_B_COLS / 4);
    const int B_view_tx = threadIdx.x % (TILE_B_COLS / 4);
    const int B_stride = n_threads_per_block / (TILE_B_COLS / 4);

    // starting indices for the registers that accumulate the dot product
    const int row_start = COLS_PER_THREAD * (threadIdx.x / (TILE_B_COLS / ROWS_PER_THREAD));
    const int col_start = ROWS_PER_THREAD * (threadIdx.x % (TILE_B_COLS / ROWS_PER_THREAD));

    // repeated memory accesses to global memory are slow -> use shared memory for tiles
    // shared memory is small -> use tiles rather than the whole matrix
    __shared__ float A_shared[TILE_A_COLS][TILE_A_ROWS];
    __shared__ float B_shared[TILE_A_COLS][TILE_B_COLS];

    const int num_tiles = ceil(A_n_cols / (float) TILE_A_COLS);

    // accumulate COLS_PER_THREADxROWS_PER_THREAD values in registers before writing to global memory
    float value[COLS_PER_THREAD][ROWS_PER_THREAD] = {0.0f};

    float A_temp[ROWS_PER_THREAD] = {0.0f};
    float B_temp[COLS_PER_THREAD] = {0.0f};

    const int A_shared_start_row = A_view_tx * 4;
    const int B_shared_start_row = B_view_tx * 4;

    for (int tile = 0; tile < num_tiles; tile++) {
        // load tiles into shared memory first
        for (int offset = 0; offset < TILE_A_ROWS; offset += A_stride) {
            int shared_col = A_view_ty + offset;

            int global_row = blockIdx.y * TILE_A_ROWS + offset + A_view_ty;
            int global_col = tile * TILE_A_COLS + (A_view_tx * 4);

            if (global_row < C_n_rows && global_col < A_n_cols) {
                float4 A_tmp = reinterpret_cast<float4*>(&d_A_ptr[global_row * A_n_cols + global_col])[0];
                A_shared[A_shared_start_row][shared_col] = A_tmp.x;
                A_shared[A_shared_start_row + 1][shared_col] = A_tmp.y;
                A_shared[A_shared_start_row + 2][shared_col] = A_tmp.z;
                A_shared[A_shared_start_row + 3][shared_col] = A_tmp.w;
            } else {
                A_shared[A_shared_start_row][shared_col] = 0.0f;
                A_shared[A_shared_start_row + 1][shared_col] = 0.0f;
                A_shared[A_shared_start_row + 2][shared_col] = 0.0f;
                A_shared[A_shared_start_row + 3][shared_col] = 0.0f;
            }
        }

        for (int offset = 0; offset < TILE_A_COLS; offset += B_stride) {
            int global_row = tile * TILE_A_COLS + B_view_ty + offset;
            int global_col = blockIdx.x * TILE_B_COLS + (B_view_tx * 4);

            if (global_row < A_n_cols && global_col < C_n_cols) {
                float4 B_tmp = reinterpret_cast<float4*>(&d_B_ptr[global_row * C_n_cols + global_col])[0];
                B_shared[B_view_ty + offset][B_shared_start_row] = B_tmp.x;
                B_shared[B_view_ty + offset][B_shared_start_row + 1] = B_tmp.y;
                B_shared[B_view_ty + offset][B_shared_start_row + 2] = B_tmp.z;
                B_shared[B_view_ty + offset][B_shared_start_row + 3] = B_tmp.w;
            } else {
                B_shared[B_view_ty + offset][B_shared_start_row] = 0.0f;
                B_shared[B_view_ty + offset][B_shared_start_row + 1] = 0.0f;
                B_shared[B_view_ty + offset][B_shared_start_row + 2] = 0.0f;
                B_shared[B_view_ty + offset][B_shared_start_row + 3] = 0.0f;
            }
        }

        // sync ensures all threads have loaded tiles into shared memory
        __syncthreads();

        for (int j = 0; j < TILE_A_COLS; j++) {
            // load the block into registers, avoiding repeated shared memory accesses
            for (int i = 0; i < ROWS_PER_THREAD; i++) {
                A_temp[i] = A_shared[j][row_start + i];
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
                int idx = global_row * C_n_cols + global_col;
                if constexpr(AlphaOne && BetaZero) {
                    d_C_ptr[idx] = value[y][x];
                } else if constexpr(AlphaOne) {
                    d_C_ptr[idx] = value[y][x] + beta * d_C_ptr[idx];
                } else if constexpr(BetaZero) {
                    d_C_ptr[idx] = alpha * value[y][x];
                } else {
                    d_C_ptr[idx] = alpha * value[y][x] + beta * d_C_ptr[idx];
                }
            }
        }
    }
}

void matmul(torch::Tensor a, torch::Tensor b, torch::Tensor out, float alpha = 1.0f, float beta = 0.0f) {
    int C_n_rows = a.size(0);
    int C_n_cols = b.size(1);
    int A_n_cols = a.size(1);

    dim3 dim_block(TILE_A_ROWS * TILE_B_COLS / (ROWS_PER_THREAD * COLS_PER_THREAD));
    dim3 dim_grid(ceil(C_n_cols / (float)(TILE_B_COLS)), ceil(C_n_rows / (float)(TILE_A_ROWS)));

    constexpr float eps = 1e-6f;
    bool alpha_one = fabs(alpha - 1.0f) < eps;
    bool beta_zero = fabs(beta) < eps;

    if (alpha_one && beta_zero) {
        matmul_kernel<true, true><<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols, alpha, beta);
    } else if (alpha_one) {
        matmul_kernel<true, false><<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols, alpha, beta);
    } else if (beta_zero) {
        matmul_kernel<false, true><<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols, alpha, beta);
    } else {
        matmul_kernel<false, false><<<dim_grid, dim_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), C_n_rows, C_n_cols, A_n_cols, alpha, beta);
    }
}
