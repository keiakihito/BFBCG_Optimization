#ifndef CHECKS_H
#define CHECKS_H

#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

// cuBLAS error string function
const char* cublasGetErrorString(cublasStatus_t err) {
    switch (err) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "UNKNOWN_CUBLAS_ERROR";
    }
}

// cuSolver error string function
const char* cusolverGetErrorString(cusolverStatus_t err) {
    switch (err) {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        default:
            return "UNKNOWN_CUSOLVER_ERROR";
    }
}

// cuSPARSE error string function
const char* cusparseGetErrorString(cusparseStatus_t err) {
    switch (err) {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        default:
            return "UNKNOWN_CUSPARSE_ERROR";
    }
}

// Error checking macros for cudaMalloc
#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

// Error checking macros for cublasStatus_t
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %s\n", __FILE__, __LINE__, cublasGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Error checking macros for cusolverStatus_t
#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSolver error at %s:%d: %s\n", __FILE__, __LINE__, cusolverGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Error checking macros for cusparseStatus_t
#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d: %s\n", __FILE__, __LINE__, cusparseGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CHECK_MAGMA(call) do { \
    magma_int_t err = call; \
    if (err != MAGMA_SUCCESS) { \
        fprintf(stderr, "MAGMA error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while (0)


#endif // CHEKS_H
