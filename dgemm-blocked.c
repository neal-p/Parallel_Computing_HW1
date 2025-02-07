#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Optimized blocked dgemm with SIMD";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

// Print matrix stored in column-major order.
void print_matrix(const double *matrix, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            printf("%10.4f ", matrix[i + j * n_rows]);
        }
        printf("\n");
    }
}

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {


    // Repack A and B so there is no striding after initial access
    // reverting back to non-transposed B for now
    double A_pack[M * K];
    double B_pack[K * N];

    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++) {
            A_pack[i + k * M] = A[i + k * lda];
        }
    }

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B_pack[k + j * K] = B[k + j * lda];
        }
    }

   // Print and make sure this is correct
  
   //printf("\nA_pack:\n"); 
   //print_matrix(A_pack, M, K);

   //printf("\nB_pack_trans:\n"); 
   //print_matrix(B_pack, K, N);


    // Update everything to use M N K lda  from my aliases
    int n_chunks = M / 4;
    int remainder = M % 4;
    for (int j = 0; j < N; j++) {
        for (int chunk = 0; chunk < n_chunks; chunk++) {

            int i = chunk * 4;
            __m256d c_vec = _mm256_loadu_pd(&C[i + j * lda]);

            for (int k = 0; k < K; k++) {

                __m256d a_vec = _mm256_loadu_pd(&A_pack[i + k * M]);
                __m256d b_val = _mm256_broadcast_sd(&B_pack[k + j * K]);
                c_vec = _mm256_fmadd_pd(a_vec, b_val, c_vec);
            }

            _mm256_storeu_pd(&C[i + j * lda], c_vec);
        }

        for (int r = 0; r < remainder; r++) {

            int i = n_chunks * 4 + r;
            double sum = 0.0;

            for (int k = 0; k < K; k++) {
                sum += A_pack[i + k * M] * B_pack[k + j * K];
            }

            C[i + j * lda] += sum;
        }
    }
}

/*
  square_dgemm performs the matrix-matrix multiplication
    C := C + A * B,
  where A, B, and C are lda-by-lda matrices stored in column-major order.
  The matrix is blocked by BLOCK_SIZE; if the matrix dimensions are not an exact multiple
  of BLOCK_SIZE, the partial blocks are handled correctly.
*/
void square_dgemm(int lda, double* A, double* B, double* C) {
    //printf("A:\n");
    //print_matrix(A, lda, lda);
    //printf("\nB:\n");
    //print_matrix(B, lda, lda);

    // Loop over the blocks.
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Compute the actual dimensions for this block.
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);

                //printf("Block: i=%i, j=%i, k=%i | M=%i, N=%i, K=%i\n", i, j, k, M, N, K);
                do_block(lda, M, N, K, 
                                        A + i + k * lda,   
                                        B + k + j * lda,   
                                        C + i + j * lda);  
            }
        }
    }
}

