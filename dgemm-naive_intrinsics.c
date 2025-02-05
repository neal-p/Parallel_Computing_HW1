const char* dgemm_desc = "Naive, three-loop dgemm.";


#include <immintrin.h>

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */


//#define PADDED

#include <stdio.h>
#include <immintrin.h>

// void square_dgemm(int n, double* A, double* B, double* C) {

//   int n_chunks = n / 4; // integer division with truncation is ok
//   int remainder = n % 4;

//   // Loop over all emements of B
//   for (int B_col=0; B_col < n; ++B_col) {
//       for (int B_row=0; B_row < n; ++B_row) {

//         // Flat index and val of element in B
//         int B_flat_idx = (B_col * n) + B_row;
//         double B_val = B[B_flat_idx];

//         printf("B_flat_idx: %i\n", B_flat_idx);

//         // Broadcast so it is repeated 4 times
//         __m256d B_val_broad = _mm256_set1_pd(B_val);


//         for (int chunk=0; chunk < n_chunks; chunk++) {

//             // Get the corresponding COL of A
//             int A_flat_idx_col_start = (B_row * n) + (chunk * 4);

//             printf("    A_flat_idx_col_start: %i\n", A_flat_idx_col_start);

//             __m256d A_col = _mm256_loadu_pd(&A[A_flat_idx_col_start]);
//             
//             // Get the result COL of C
//             int C_flat_idx_col_start = (B_col * n) + (chunk * 4);

//             printf("    C_flat_idx_col_start: %i\n", C_flat_idx_col_start);

//             __m256d C_col = _mm256_loadu_pd(&C[C_flat_idx_col_start]);

//             // Multiply the B_val by the COL of A, add to COL of C
//             C_col = _mm256_fmadd_pd(B_val_broad, A_col, C_col);

//             // Store back in C
//             _mm256_storeu_pd(&C[C_flat_idx_col_start], C_col);
//           }

//         if (remainder > 0) {

//             #ifdef PADDED
//             // Try padding in a temp array
//             // I wonder if this is actually slower bc it needs to 
//             // create a two temp arrays

//             int A_flat_idx_col_start = (B_row * n) + (n_chunks * 4);
//             int C_flat_idx_col_start = (B_col * n) + (n_chunks * 4);

//             double A_padded[4] = {0.0, 0.0, 0.0, 0.0};
//             double C_padded[4] = {0.0, 0.0, 0.0, 0.0};

//             for (int r=0; r < remainder; ++r) {
//                 A_padded[r] = A[A_flat_idx_col_start + r];
//                 C_padded[r] = C[C_flat_idx_col_start + r];
//             }

//             __m256d A_col = _mm256_loadu_pd(A_padded);
//             __m256d C_col = _mm256_loadu_pd(C_padded);

//             // Multiply the B_val by the COL of A, add to COL of C
//             C_col = _mm256_fmadd_pd(B_val_broad, A_col, C_col);

//             // Store back in C_padded
//             _mm256_storeu_pd(C_padded, C_col);

//             // Put back in real C
//              for (int r=0; r < remainder; ++r) {
//                 C[C_flat_idx_col_start + r] = C_padded[r];
//             }


//             #else 
//             // Try just serially doing the last 1-3 values
//             // even tho it isn't vectorized I'm guessing it'll be
//             // faster than padding???
//         
//             int A_flat_idx_col_start = (B_row * n) + (n_chunks * 4);
//             int C_flat_idx_col_start = (B_col * n) + (n_chunks * 4);

//             for (int r=0; r < remainder; ++r) {
//                 C[C_flat_idx_col_start + r] = B_val * A[A_flat_idx_col_start + r] + C[C_flat_idx_col_start + r];
//             }

//             #endif

//       }
//     }
//   }
// }





void square_dgemm(int n, double* A, double* B, double* C) {

  for (int A_col=0; A_col < n; ++A_col) {

    for ()



  }





}









