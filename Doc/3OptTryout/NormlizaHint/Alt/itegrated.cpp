#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macros (Assuming they are defined elsewhere)
#define CHECK_CUDA(call)  // Error checking for CUDA calls
#define CHECK_CUBLAS(call) // Error checking for cuBLAS calls

// Kernel to compute inverses of norms
__global__ void compute_inverses(double* norms, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (norms[idx] > 0.0) {
            norms[idx] = 1.0 / norms[idx];
        } else {
            norms[idx] = 0.0; // Handle zero norm to avoid division by zero
        }
    }
}

// Function to normalize the columns of mtxY_d
void normalize_Den_Mtx(
    cublasHandle_t cublasHandler,
    double* mtxY_d,
    int numRows,
    int numCols)
{
    // Allocate device memory for norms
    double* norms_d = NULL;
    CHECK_CUDA(cudaMalloc((void**)&norms_d, numCols * sizeof(double)));

    // Compute norms of each column (device computation)
    for (int i = 0; i < numCols; ++i) {
        CHECK_CUBLAS(cublasDnrm2(
            cublasHandler,
            numRows,
            mtxY_d + i * numRows, // Pointer to column i
            1,                    // Increment (since columns are contiguous in memory)
            norms_d + i));        // Store norm in norms_d[i]
    }

    // Compute inverses of norms using a custom kernel
    int blockSize = 256;
    int gridSize = (numCols + blockSize - 1) / blockSize;
    compute_inverses<<<gridSize, blockSize>>>(norms_d, numCols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Scale each column of mtxY_d using the inverses
    for (int i = 0; i < numCols; ++i) {
        CHECK_CUBLAS(cublasDscal(
            cublasHandler,
            numRows,
            norms_d + i,           // Pointer to the inverse norm for column i
            mtxY_d + i * numRows,  // Pointer to column i
            1));                   // Increment
    }

    // Free device memory
    CHECK_CUDA(cudaFree(norms_d));
}
