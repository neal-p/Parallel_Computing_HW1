#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

const char* dgemm_desc = "Optimized blocked dgemm with SIMD";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
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

void single_dot_product(__m256d v1, __m256d v2, double* res) {

    // Multiply
    __m256d mul = _mm256_mul_pd(v1, v2);
   
    // Horizontal sum from https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
    __m128d vlow  = _mm256_castpd256_pd128(mul);
    __m128d vhigh = _mm256_extractf128_pd(mul, 1);
    vlow  = _mm_add_pd(vlow, vhigh);
    
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);

    // Store in res
    *res = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

void double_dot_product(__m256d v1, __m256d v2, __m256d v3, __m256d v4, double* res1, double* res2) {

    // Multiply
    __m256d mul1 = _mm256_mul_pd(v1, v2);
    __m256d mul2 = _mm256_mul_pd(v3, v4);
   
    // Horizontal sum from https://stackoverflow.com/questions/9775538/fastest-way-to-do-horizontal-vector-sum-with-avx-instructions
    

    __m256d sum = _mm256_hadd_pd(mul1, mul2);
    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
    __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));

    // Store in repsective res's
    *res1 = _mm_cvtsd_f64(result);
    *res2 = _mm_cvtsd_f64(_mm_shuffle_pd(result, result, 1));

}

/*
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {

    double A_pack[M * K];
    double B_pack[K * N];

    // Pack A
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A_pack[i * K + k] = A[i + k * lda];
        }
    }

    // Pack B
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B_pack[k + j * K] = B[k + j * lda];
        }
    }

    //printf("\nA_pack:\n");
    //print_matrix(A_pack, 1, M*K); // print flat to be super sure of memory layout

    //printf("\nB_pack:\n");
    //print_matrix(B_pack, 1, K*N); // print flat to be super sure of memory layout

    int n_chunks = K / 4;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {

            double dp_sum = 0.0;

	    // If I could do two chunks at a time I think
	    // the horizontal sum might be more efficient with
	    // two at a time
	    // https://stackoverflow.com/questions/9775538/fastest-way-to-do-horizontal-vector-sum-with-avx-instructions
            for (int chunk = 0; chunk < n_chunks; ++chunk) {
                int A_index = i * K + (4 * chunk);
                int B_index = j * K + (4 * chunk);

                __m256d A_chunk = _mm256_loadu_pd(&A_pack[A_index]);
                __m256d B_chunk = _mm256_loadu_pd(&B_pack[B_index]);

                dp_sum += dot_product(A_chunk, B_chunk);
            }

            // Handle remainders
            for (int r = n_chunks * 4; r < K; r++) {
                dp_sum += A_pack[i * K + r] * B_pack[j * K + r];
            }

            C[i + j * lda] += dp_sum; // remember to use lda since C isnt packed!!!!
        }
    }
}



*/

static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {

    //double A_pack[M * K];
    //double B_pack[K * N];

    double* A_pack = (double*)_mm_malloc(M * K * sizeof(double), BLOCK_SIZE);
    double* B_pack = (double*)_mm_malloc(K * N * sizeof(double), BLOCK_SIZE);

    // Pack A
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A_pack[i * K + k] = A[i + k * lda];
        }
    }

    // Pack B
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B_pack[k + j * K] = B[k + j * lda];
        }
    }

    //printf("\nA_pack:\n");
    //print_matrix(A_pack, 1, M*K); // print flat to be super sure of memory layout

    //printf("\nB_pack:\n");
    //print_matrix(B_pack, 1, K*N); // print flat to be super sure of memory layout

    int n_chunks = K / 8;
    int single_n_chunks = (K - (n_chunks * 8)) / 4;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {

            double dp_sum = 0.0;

	    // If I could do two chunks at a time I think
	    // the horizontal sum might be more efficient with
	    // two at a time
	    // https://stackoverflow.com/questions/9775538/fastest-way-to-do-horizontal-vector-sum-with-avx-instructions
            for (int chunk = 0; chunk < n_chunks; ++chunk) {

		 //printf("Double dot product\n");
                int A_index = i * K + (8 * chunk);
                int B_index = j * K + (8 * chunk);

                __m256d A_chunk1 = _mm256_loadu_pd(&A_pack[A_index]);
                __m256d B_chunk1 = _mm256_loadu_pd(&B_pack[B_index]);

		__m256d A_chunk2 = _mm256_loadu_pd(&A_pack[A_index+4]);
		__m256d B_chunk2 = _mm256_loadu_pd(&B_pack[B_index+4]);

		double res1, res2;
		double_dot_product(A_chunk1, B_chunk1, A_chunk2, B_chunk2, &res1, &res2);

		dp_sum += res1 + res2;
            }

            // Handle remainders

            // single chunks

	    for (int single_chunk=0; single_chunk < single_n_chunks; ++single_chunk) {

		    
		 //printf("single dot product\n");

                int A_index = i * K + (8 * n_chunks) + (4 * single_chunk);
                int B_index = j * K + (8 * n_chunks) + (4 * single_chunk);

                __m256d A_chunk1 = _mm256_loadu_pd(&A_pack[A_index]);
                __m256d B_chunk1 = _mm256_loadu_pd(&B_pack[B_index]);

		double res1;
		single_dot_product(A_chunk1, B_chunk1, &res1);

		dp_sum += res1;
	    }


	    // final scalars left
	    
            for (int r = (n_chunks * 8) + (single_n_chunks * 4); r < K; r++) {


		 //printf("scalar product\n");
		   
                dp_sum += A_pack[i * K + r] * B_pack[j * K + r];
            }

            C[i + j * lda] += dp_sum; // remember to use lda since C isnt packed!!!!
        }
    }

    _mm_free(A_pack);
    _mm_free(B_pack);

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
   // printf("\nB:\n");
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

