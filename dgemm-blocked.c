#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
const char* dgemm_desc = "Optimized blocked dgemm with SIMD";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8  // TODO: Tuned block size for better cache performance
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))



void print_matrix(const double *matrix, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            printf("%10.4f ", matrix[j * n_rows + i]); // Column-major indexing
        }
        printf("\n");
    }
}





/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

/*
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}
*/


//* where C is M-by-N, A is M-by-K, and B is K-by-N.
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {


   int A_block_rows =  K;
   int A_block_cols = M;
   int B_block_rows = N;
   int B_block_cols = K;

   int full_matrix_cols = lda;
   int full_matrix_rows = lda;


   // Repack B so there is no striding after initial access
   double B_pack_trans[B_block_rows * B_block_cols];

   int B_pack_idx = 0;
   for (int B_row=0; B_row < B_block_rows; ++B_row) {
	   for (int B_col=0; B_col < B_block_cols; ++B_col) {

		   int B_flat_idx = (B_col * full_matrix_cols) + B_row;
		   B_pack_trans[B_pack_idx] = B[B_flat_idx];
		   B_pack_idx += 1;
	   }
   }


   // Repack A so there is no striding after initial access
   double A_pack[A_block_rows * A_block_cols];

   int A_pack_idx = 0;
   for (int A_col=0; A_col < A_block_cols; ++A_col) {
	   for (int A_row=0; A_row < A_block_rows; ++A_row) {

		   int A_flat_idx = (A_col * full_matrix_cols) + A_row;
		   A_pack[A_pack_idx] = A[A_flat_idx];
		   A_pack_idx += 1;
	   }
   }


   // Print and make sure this is correct
  
   // printf("\nA_pack:\n"); 
   // print_matrix(A_pack, A_block_cols, A_block_rows);

   // printf("\nB_pack_trans:\n"); 
   // print_matrix(B_pack_trans, B_block_rows, A_block_cols);


   int remainder = A_block_rows % 4;
   int n_chunks = A_block_rows / 4;

   printf("A_block_rows=%i\n", A_block_rows);
   printf("n_chunks=%i\n", n_chunks);
   printf("remainder=%i\n", remainder);



   for (int j=0; j < B_block_cols; ++j) {

	   for (int chunk=0; chunk < n_chunks; ++chunk) {
		   int i = chunk * 4;
		   printf("AVX section i=%i\n", i);

		   __m256d c_vec = _mm256_loadu_pd(&C[(j * full_matrix_cols) + i]);

		   for (int k=0; k < A_block_cols; ++k) {

			   __m256d a_vec = _mm256_loadu_pd(&A_pack[(k * A_block_rows) + i]);
			   __m256d b_val = _mm256_broadcast_sd(&B_pack_trans[(k * B_block_cols) + j]);

			   c_vec = _mm256_fmadd_pd(a_vec, b_val, c_vec);
		   }

		   _mm256_storeu_pd(&C[(j * full_matrix_cols) + i], c_vec);
	   }


	   for (int r=0; r < remainder; ++r) {

		   int i = (n_chunks * 4) + r;

		   printf("scalar section i=%i\n", i);

		   double sum = 0.0;

		   for (int k=0; k < A_block_cols; ++k) {
			   sum += A_pack[(k * A_block_rows) + i] * B_pack_trans[(k* B_block_cols) + j];
		   }

		   C[(j * full_matrix_cols) + i] += sum;
	   }

   }

}







/*
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {

    //double B_trans[K * N]; // Stack allocation for small sizes

    // Transpose B into B_transposed (row-major layout)
    //for (int j = 0; j < N; ++j) {
    //    for (int k = 0; k < K; ++k) {
    //        B_trans[j * K + k] = B[k + j * lda]; // Swap indices
    //    }
    //}


    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
	    double cij = C[i + j * lda];

            for (int k = 0; k < K; ++k) {
		double aik = A[i + k * lda];
		//double bkj = B_trans[k + j * K];
		double bkj = B[j + k*K];

		cij += aik * bkj;
            }

	    C[i + j * lda] = cij;
        }
    }
}

*/



/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {



    printf("A:\n"); 
    print_matrix(A, lda, lda);

    printf("\nB:\n"); 
    print_matrix(B, lda, lda);



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


		printf("i=%i, j=%i, k=%i | M=%i, N=%i, K=%i | A start: %i, B start: %i, C start: %i\n\n",
				 i,j,k,
				 M,N,K,
				 i + k *lda, k + j * lda, i + j * lda);
 

                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);

            }
	}
    }
}
