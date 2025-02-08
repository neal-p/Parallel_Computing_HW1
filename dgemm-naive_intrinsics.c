#include <stdio.h>
#include <immintrin.h>

const char* dgemm_desc = "Naive, three-loop dgemm.";


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

                __m256d A_col = _mm256_loadu_pd(&A[A_flat_idx]);
                __m256d C_col = _mm256_loadu_pd(&C[C_flat_idx]);
                C_col = _mm256_fmadd_pd(B_val_broad, A_col, C_col);
                _mm256_storeu_pd(&C[C_flat_idx], C_col);
            }

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


void square_dgemm(int n, double* A, double* B, double* C) {
    // Tile sizes for B_col (j dimension) and B_row (k dimension)
    int BLOCK_J = 128;  // Adjust as needed for your cache size
    int BLOCK_K = 128;  // Adjust as needed

    for (int j = 0; j < n; j += BLOCK_J) {
        int j_end = (j + BLOCK_J < n) ? (j + BLOCK_J) : n;
        for (int k = 0; k < n; k += BLOCK_K) {
            int k_end = (k + BLOCK_K < n) ? (k + BLOCK_K) : n;
            dgemm_block(n, j, j_end, k, k_end, A, B, C);
        }
    }
}




/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */

/*
void square_dgemm(int n, double* A, double* B, double* C) {

  int n_chunks = n / 4; // integer division with truncation is ok
  int remainder = n % 4;

  // Loop over all emements of B
  for (int B_col=0; B_col < n; ++B_col) {



      for (int B_row=0; B_row < n; ++B_row) {

        // Flat index and val of element in B
        int B_flat_idx = (B_col * n) + B_row;
        double B_val = B[B_flat_idx];

        // Broadcast so it is repeated 4 times
        __m256d B_val_broad = _mm256_set1_pd(B_val);


        for (int chunk=0; chunk < n_chunks; chunk++) {


            // Get the corresponding COL of A
            int A_flat_idx_col_start = (B_row * n) + (chunk * 4);

            __m256d A_col = _mm256_loadu_pd(&A[A_flat_idx_col_start]);
            
            // Get the result COL of C
            int C_flat_idx_col_start = (B_col * n) + (chunk * 4);

            __m256d C_col = _mm256_loadu_pd(&C[C_flat_idx_col_start]);

            // Multiply the B_val by the COL of A, add to COL of C
            C_col = _mm256_fmadd_pd(B_val_broad, A_col, C_col);

            // Store back in C
            _mm256_storeu_pd(&C[C_flat_idx_col_start], C_col);


            //printf("B_flat_idx: %i\n", B_flat_idx);
            //printf("A_flat_idx: %i\n", A_flat_idx_col_start);
            //printf("C_flat_idx: %i\n", C_flat_idx_col_start);
            //printf("\n");

          }

        _mm_prefetch((char*)&A[0], _MM_HINT_T0);

        if (remainder > 0) {

            int A_flat_idx_col_start = (B_row * n) + (n_chunks * 4);
            int C_flat_idx_col_start = (B_col * n) + (n_chunks * 4);

            for (int r=0; r < remainder; ++r) {
                C[C_flat_idx_col_start + r] = B_val * A[A_flat_idx_col_start + r] + C[C_flat_idx_col_start + r];
            }
        }

    }
  }
}

*/

