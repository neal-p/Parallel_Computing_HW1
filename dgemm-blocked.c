#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
const char* dgemm_desc = "Optimized blocked dgemm with SIMD";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    int M4 = M / 4 * 4;  // round M down to nearest multiple of 4
    int K4 = K / 4 * 4;  // round K down to nearest multiple of 4
    int N4 = N / 4 * 4;  // round K down to nearest multiple of 4
    double A_local[BLOCK_SIZE * BLOCK_SIZE], B_local[BLOCK_SIZE*BLOCK_SIZE];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A_local[i + j*BLOCK_SIZE] = A[i + j * lda];
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B_local[i + j*BLOCK_SIZE] = B[i + j * lda];
        }
    }

    for (int j = 0; j < N4; j+=4) { 
        for (int i = 0; i < M4; i += 4) { 
            // load 4 double-precision floating point values, u - unaligned memory
            __m256d c0 = _mm256_loadu_pd(&C[i + (j+0) * lda]);
            __m256d c1 = _mm256_loadu_pd(&C[i + (j+1) * lda]);
            __m256d c2 = _mm256_loadu_pd(&C[i + (j+2) * lda]);
            __m256d c3 = _mm256_loadu_pd(&C[i + (j+3) * lda]);

            // for (int k = 0; k < K4; k += 4) {
            for (int k = 0; k < K4; k++) {
                __m256d a0 = _mm256_loadu_pd(A_local + i + k * BLOCK_SIZE);

                // duplicates a single double value across all 4 lanes
                // used when multiplying a row of A with a single value from B, more efficient than loading the same value 4 times 
                __m256d b0 = _mm256_broadcast_sd(B_local + k + j * BLOCK_SIZE);
                __m256d b1 = _mm256_broadcast_sd(B_local + k + (j+1) * BLOCK_SIZE);
                __m256d b2 = _mm256_broadcast_sd(B_local + k + (j+2) * BLOCK_SIZE);
                __m256d b3 = _mm256_broadcast_sd(B_local + k + (j+3) * BLOCK_SIZE);

                // perform fused multiply-add, (A * B) + C
                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
            }

            // Handle remaining elements if K is not a multiple of 4
            for (int k_rem = K4; k_rem < K; ++k_rem) {
                __m256d a0 = _mm256_loadu_pd(&A[i + k_rem * lda]);
                __m256d b0 = _mm256_broadcast_sd(&B[k_rem + j * lda]);
                __m256d b1 = _mm256_broadcast_sd(&B[k_rem + (j+1) * lda]);
                __m256d b2 = _mm256_broadcast_sd(&B[k_rem + (j+2) * lda]);
                __m256d b3 = _mm256_broadcast_sd(&B[k_rem + (j+3) * lda]);
                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
                
            }

            // stores 4 double-precision floating-point values back into memory
            _mm256_storeu_pd(&C[i + j * lda], c0);
            _mm256_storeu_pd(&C[i + (j+1) * lda], c1);
            _mm256_storeu_pd(&C[i + (j+2) * lda], c2);
            _mm256_storeu_pd(&C[i + (j+3) * lda], c3);
        }
    }

    // Handle Remaining Columns (N % 4 != 0)
    for (int j = N4; j < N; ++j) {
        for (int i = 0; i < M4; i += 4) {
            __m256d c0 = _mm256_loadu_pd(&C[i + j * lda]);
            for (int k = 0; k < K; ++k) {
                __m256d a0 = _mm256_loadu_pd(&A[i + k * lda]);
                __m256d b0 = _mm256_broadcast_sd(&B[k + j * lda]);
                c0 = _mm256_fmadd_pd(a0, b0, c0);
            }
            _mm256_storeu_pd(&C[i + j * lda], c0);
        }
    }

    // Handle Remaining Rows (M % 4 != 0)
    for (int j = 0; j < N; ++j) {
        for (int i = M4; i < M; ++i) {
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);

                // Perform individual block dgemm
                do_block(lda, M, N, K, A + k * lda + i, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}