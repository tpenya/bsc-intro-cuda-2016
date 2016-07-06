/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Kernel of dense matrix-matrix multiplication kernel.
 * The algorithm is based on CUDA sgemm code from Vasily Volkov
 * at UC Berkeley.
 */

#include <iostream>

#include "sgemm_kernel.h"

// Returns x/y ceiled to the next (upper) integer when x doesn't evenly divide by y (uses integer arithmetic trickery)
#define div_and_ceil(x,y) (((x) - 1)/(y) + 1)

// Select a kernel implementation. Comment/uncomment the line below to switch between the simple and tiled versions
#define SIMPLE

// Parameters of tile sizes (only needed for the tiled implementation)
#define TILE_SZ_M 128
#define TILE_SZ_N 16
#define TILE_SZ_K (TILE_SZ_M/TILE_SZ_N) // keep this ratio to ease B_s loading (i.e. # of elements in B_s == # of threads in a thread block)

#ifdef SIMPLE
// simple sgemm kernel implementation.
// Note: A and C are stored in memory in column major order, and B is stored in row major.
__global__ void sgemmNT_naive( const float *A, const float *B, float* C, int m, int n, int k )
{
    float c = 0.0f;
    int im = blockIdx.x * blockDim.x + threadIdx.x;
    int in = blockIdx.y * blockDim.y + threadIdx.y;
    for (int ik = 0; ik < k; ++ik) {
        float a = A[im + ik * m];
        float b = B[in + ik * n];
        c += a * b;
    }
    C[im + in*m] = c;
}

#else

// sgemm kernel implementation with shared memory and register tiling.
// Note: A and C are stored in memory in column major order, and B is stored in row major.
__global__ void sgemmNT_tiled( const float *A, const float *B, float* C, int m, int n, int k )
{
    // Shared memory allocation to store a tile of B
    __shared__ float B_s [TILE_SZ_K][TILE_SZ_N];

    // Macros for accessing flattened matrices
    #define A(row,col) A[row + (col)*m]
    #define B(row,col) B[(row)*n + (col)]
    #define C(row,col) C[row + (col)*m]

    // Compute thread's global row index (for A and C)
    const unsigned int im = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute the global column index of the first column processed by the thread (for B and C)   
    const unsigned int in = blockIdx.y * TILE_SZ_N;

    // Privatization of output variables. Each thread computes a row of a tile of C.
    float c_reg[TILE_SZ_N];

    // Initialize output values
    for(unsigned int outIdx = 0; outIdx < TILE_SZ_N; ++outIdx) {
        c_reg[outIdx] = 0;
    }

    // Loop over the input tiles following the K dimension
    for(unsigned int tileIdx = 0; tileIdx < div_and_ceil(k,TILE_SZ_K); ++tileIdx) {
        // Compute the coordinates of the element of B_s loaded by each thread 
        const unsigned int iB = threadIdx.x / TILE_SZ_N;
        const unsigned int jB = threadIdx.x % TILE_SZ_N;
        // Load the current tile of B into shared memory. Ensure all threads finished before proceeding.
        if(tileIdx * TILE_SZ_K + iB < k && in + jB < n) {
            B_s[iB][jB] = B(tileIdx * TILE_SZ_K + iB, in + jB);
        } else {
            B_s[iB][jB] = 0;
        }
        __syncthreads();
        // Loop over the columns of A's tile and the rows of B's tile
        for(unsigned int idx = 0; idx < TILE_SZ_K; ++idx) {
            // Load current element of A matrix into the register
            float a_reg;
            if(im < m && tileIdx * TILE_SZ_K + idx < k) {
                a_reg = A(im, tileIdx * TILE_SZ_K + idx);
            } else {
                a_reg = 0;
            }
            // Loop over the columns of B_s and update the output elements assigned to the thread
            for(unsigned int outIdx = 0; outIdx < TILE_SZ_N; ++outIdx) {
                c_reg[outIdx] += a_reg * B_s[idx][outIdx];
            }
        }
        // Ensure all threads finished before proceeding.
        __syncthreads();
    }
    
    // Store the result to C
    for(unsigned int outIdx = 0; outIdx < TILE_SZ_N; ++outIdx) {
        if(im < m && in + outIdx < n) {
            C(im, in + outIdx) = c_reg[outIdx];
        }
    }
}
#endif

void sgemm( const float *A, const float *B, float* C, int m, int n, int k )
{
#ifdef SIMPLE
    std::cout << std::endl << "Using the SIMPLE kernel implementation" << std::endl;

    const unsigned block_sz = 16;

    dim3 grid( m/block_sz, n/block_sz ), threads( block_sz, block_sz );
    sgemmNT_naive<<<grid, threads>>>( A, B, C, m, n, k);

#else
    std::cout << std::endl << "Using the TILED kernel implementation" << std::endl;

    dim3 grid(div_and_ceil(m,TILE_SZ_M), div_and_ceil(n,TILE_SZ_N)), threads(TILE_SZ_M, 1);
    sgemmNT_tiled<<<grid, threads>>>( A, B, C, m, n, k);

#endif
    CHECK_ERROR();
}

