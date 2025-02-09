#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

const char* dgemm_desc = "Naive, three-loop dgemm.";

// True if memory is not aligned
// AND is alignable
static inline int need_aligned(const void* ptr) {
    return (((uintptr_t)ptr % BLOCK_SIZE) != 0);
}

static void dgemm_block(int n, int j_start, int j_end, int k_start, int k_end,
                        double *A, double *B, double *C) {
    int n_chunks = n / 4;
    int remainder = n % 4;

    for (int j = j_start; j < j_end; ++j) {       
        for (int k = k_start; k < k_end; ++k) {     
            int B_flat_idx = j * n + k;
            double B_val = B[B_flat_idx];

            __m256d B_val_broad = _mm256_set1_pd(B_val);

            for (int chunk = 0; chunk < n_chunks; chunk++) {
                int i = chunk * 4;
                int A_flat_idx = k * n + i;  
                int C_flat_idx = j * n + i; 

		        if (!need_aligned(&A[A_flat_idx])) {
                    __m256d A_col = _mm256_load_pd(&A[A_flat_idx]);
                    __m256d C_col = _mm256_load_pd(&C[C_flat_idx]);
                    C_col = _mm256_fmadd_pd(B_val_broad, A_col, C_col);
                    _mm256_store_pd(&C[C_flat_idx], C_col);
		        } else {
                    __m256d A_col = _mm256_loadu_pd(&A[A_flat_idx]);
                    __m256d C_col = _mm256_loadu_pd(&C[C_flat_idx]);
                    C_col = _mm256_fmadd_pd(B_val_broad, A_col, C_col);
                    _mm256_storeu_pd(&C[C_flat_idx], C_col);
		        }
            }

            _mm_prefetch((char*)&A[0], _MM_HINT_T0);

            if (remainder > 0) {
                int i = n_chunks * 4;
                int A_flat_idx = k * n + i;
                int C_flat_idx = j * n + i;
                for (int r = 0; r < remainder; ++r) {
                    C[C_flat_idx + r] += B_val * A[A_flat_idx + r];
                }
            }
        }
    }
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