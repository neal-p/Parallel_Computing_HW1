#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

const char* dgemm_desc = "Optimized with 8×8 Kernel";

// True if memory is aligned to 64 bytes
static inline int need_aligned(const void* ptr) {
    return (((uintptr_t)ptr % BLOCK_SIZE) != 0);
}

double* repack(double* M, int n) {
    if (!need_aligned(M)) {
        return M;
    }
    int size = ((((n*n) / BLOCK_SIZE) + 1) * BLOCK_SIZE);

    double* aligned = aligned_alloc(BLOCK_SIZE, size * sizeof(double));
    memcpy(aligned, M, n*n*sizeof(double));
    return aligned;
}

/*
 * Optimized 8×8 DGEMM Block Kernel with Packed A and C
 */
static void dgemm_block(int n, int j_start, int j_end, int k_start, int k_end,
                        double *A_pack, double *B, double *C_pack) {
    int aligned_n = (n / 8) * 8;
    int remainder_n = n % 8;
    int remainder_j = (j_end - j_start) % 8;

    for (int j = j_start; j + 7 < j_end; j += 8) {
        for (int k = k_start; k < k_end; ++k) {
            __m256d B_val_broad[8];
            int B_flat_idx[8];
            for (int idx = 0; idx < 8; idx++) {
                B_flat_idx[idx] = (j + idx) * n + k;
                B_val_broad[idx] = _mm256_set1_pd(B[B_flat_idx[idx]]);
            }

            for (int i = 0; i < aligned_n; i += 8) {
                int A_flat_idx = k * n + i;
                int C_flat_idx[8];

                for (int idx = 0; idx < 8; idx++) {
                    C_flat_idx[idx] = (j + idx) * n + i;
                }

                __m256d A_col0 = _mm256_loadu_pd(&A_pack[A_flat_idx]);
                __m256d A_col1 = _mm256_loadu_pd(&A_pack[A_flat_idx + 4]);

                __m256d C_col0[8], C_col1[8];
                for (int idx = 0; idx < 8; idx++) {
                    C_col0[idx] = _mm256_loadu_pd(&C_pack[C_flat_idx[idx]]);
                    C_col1[idx] = _mm256_loadu_pd(&C_pack[C_flat_idx[idx] + 4]);
                }

                for (int idx = 0; idx < 8; idx++) {
                    C_col0[idx] = _mm256_fmadd_pd(B_val_broad[idx], A_col0, C_col0[idx]);
                    C_col1[idx] = _mm256_fmadd_pd(B_val_broad[idx], A_col1, C_col1[idx]);
                }

                for (int idx = 0; idx < 8; idx++) {
                    _mm256_storeu_pd(&C_pack[C_flat_idx[idx]], C_col0[idx]);
                    _mm256_storeu_pd(&C_pack[C_flat_idx[idx] + 4], C_col1[idx]);
                }
            }

            if (remainder_n > 0) {
                for (int i = aligned_n; i < n; ++i) {
                    int A_flat_idx = k * n + i;
                    for (int idx = 0; idx < 8; idx++) {
                        int C_flat_idx = (j + idx) * n + i;
                        C_pack[C_flat_idx] += B[B_flat_idx[idx]] * A_pack[A_flat_idx];
                    }
                }
            }
        }
    }

    if (remainder_j > 0) {
        for (int j = j_end - remainder_j; j < j_end; ++j) {
            for (int k = k_start; k < k_end; ++k) {
                int B_flat_idx = j * n + k;
                double B_val = B[B_flat_idx];

                for (int i = 0; i < n; ++i) {
                    int A_flat_idx = k * n + i;
                    int C_flat_idx = j * n + i;
                    C_pack[C_flat_idx] += B_val * A_pack[A_flat_idx];
                }
            }
        }
    }
}

void square_dgemm(int n, double* A, double* B, double* C) {
    int BLOCK_J = BLOCK_SIZE;
    int BLOCK_K = BLOCK_SIZE;  
    double* A_pack;
    double* C_pack;

    // Repack matrices if n > 128
    if (n > 128) {
        A_pack = repack(A, n);
        C_pack = repack(C, n);
        
    }

    // Loop over blocks of the matrix
    for (int j = 0; j < n; j += BLOCK_J) {
        int j_end = (j + BLOCK_J < n) ? (j + BLOCK_J) : n;
        for (int k = 0; k < n; k += BLOCK_K) {
            int k_end = (k + BLOCK_K < n) ? (k + BLOCK_K) : n;
            // Call block computation for either packed or original matrices
            if (n > 128) {
                dgemm_block(n, j, j_end, k, k_end, A_pack, B, C_pack);
            } else {
                dgemm_block(n, j, j_end, k, k_end, A, B, C);
            }
        }
    }

    // Copy the results back to C if packed
    if (n > 128) {
        if (C_pack != C) {
            memcpy(C, C_pack, n * n * sizeof(double));
            free(C_pack);
        }

        if (A_pack != A) {
            free(A_pack);
        }
    }
}