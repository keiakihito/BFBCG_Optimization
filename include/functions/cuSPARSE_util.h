#ifndef CUSPARSE_UTIL_H
#define CUSPARSE_UTIL_H

#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "helper.h"



//Input: const CSRMatrix *csrMtx, double *dnsMtxB_d, int numClmB, double * dnxMtxC_d
//Process: Matrix Multiplication Sparse matrix and Dense matrix
//Output: dnsMtxC_d, dense matrix C in device
void multiply_Sprc_Den_mtx(cusparseHandle_t cusparseHandler, const CSRMatrix *csrMtx, double *dnsMtxB_d, int numClmsB, double * dnsMtxC_d);


// Input: const CSRMatrix *csrMtx, double *dnsVecX_d, double *dnsVecY_d
// Process: Matrix Multiplication Sparse matrix and Dense vector
// Output: dnsVecY_d, dense vector Y in device
void multiply_Sprc_Den_vec(cusparseHandle_t cusparseHandler, const CSRMatrix *csrMtx, double *dnsVecX_d, double *dnsVecY_d);


//Input: double *dnsMtxB_d, const CSRMatrix *csrMtx, double *dnsMtxX_d, int numClmB, double * dnxMtxC_d
//Process: perform C = C - AX
//Output: dnsMtxC_d, dense matrix C in device
void den_mtx_subtract_multiply_Sprc_Den_mtx(cusparseHandle_t cusparseHandler,double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d);

//Input:
//Process: perform vector y = y  - Ax
//Output: dnsVecY_d, dense vector C in device
void den_vec_subtract_multiplly_Sprc_Den_vec(cusparseHandle_t cusparseHandler, const CSRMatrix *csrMtx, double *dnsVecX_d, double *dnsVecY_d);

//Input
//Process: perform r = b - Ax, dot product r' * r, then square norm
//Output: double twoNorms residual
double validateCG(const CSRMatrix *csrMtx, int numOfA, double *dnsVecX_d, double* dnsVecY_d);


//Input: CSRMatrix *csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d
//Process: perform R = B - AX, then calculate the first column vector 2 norms
//Output: double twoNorms
double validateBFBCG(const CSRMatrix *csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d);





//Sparse matrix multiplicatation
//Input: const CSRMatrix *csrMtx, double *dnsMtxB_d, int numClmB, double * dnxMtxC_d
//Process: Matrix Multiplication Sparse matrix and Dense matrix
//Output: dnsMtxC_d, dense matrix C in device
void multiply_Sprc_Den_mtx(cusparseHandle_t cusparseHandler, const CSRMatrix *csrMtx, double *dnsMtxB_d, int numClmsB, double * dnsMtxC_d)
{
	int numRowsA = csrMtx->_numOfRows;
	int numClmsA = csrMtx->_numOfClms;
	int nnz = csrMtx->_numOfnz;

	double alpha = 1.0f;
	double beta = 0.0f;

	bool debug = false;


	//(1) Allocate device memoery for CSR matrix
	int	*row_offsets_d = NULL;
	int *col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) Copy values from host to device
	CHECK(cudaMemcpy(row_offsets_d, csrMtx->_row_offsets, (numRowsA+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx->_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx->_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3) Crate cuSPARSE handle and descriptors
	// cusparseHandle_t cusparseHandler;
	// cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnMatDescr_t mtxB, mtxC;

	CHECK_CUSPARSE(cusparseCreateCsr(&mtxA, numRowsA, numClmsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnMat(&mtxB, numClmsA, numClmsB, numClmsA, dnsMtxB_d, CUDA_R_64F, CUSPARSE_ORDER_COL));
	CHECK_CUSPARSE(cusparseCreateDnMat(&mtxC, numRowsA, numClmsB, numRowsA, dnsMtxC_d, CUDA_R_64F, CUSPARSE_ORDER_COL));

	//(4) Calculate buffer size of Spase by dense matrix mulply operation
    size_t bufferSize = 0;
    void *dBuffer = NULL;
	CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5)Perform sparse-dense matrix Multiplication
	CHECK_CUSPARSE(cusparseSpMM(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\n~~mtxC after cusparseSpMM~~\n\n");
		print_mtx_clm_d(dnsMtxC_d, numRowsA, numClmsB);
	}

	//(6) Free memeory and destroy descriptors
	CHECK_CUSPARSE(cusparseDestroySpMat(mtxA));
	CHECK_CUSPARSE(cusparseDestroyDnMat(mtxB));
	CHECK_CUSPARSE(cusparseDestroyDnMat(mtxC));
	// CHECK_CUSPARSE(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));

} // end of multiply_Src_Den_mtx



// Sparse matrix multiplication with a dense vector
// Input: const CSRMatrix *csrMtx, double *dnsVecX_d, double *dnsVecY_d
// Process: Matrix Multiplication Sparse matrix and Dense vector
// Output: dnsVecY_d, dense vector Y in device
void multiply_Sprc_Den_vec(cusparseHandle_t cusparseHandler, const CSRMatrix *csrMtx, double *dnsVecX_d, double *dnsVecY_d)
{
	int numRowsA = csrMtx->_numOfRows;
	int numColsA = csrMtx->_numOfClms;
	int nnz = csrMtx->_numOfnz;

	double alpha = 1.0f;
	double beta = 0.0f;
	
	bool debug = false;

	//(1) Allocate device memory for CSR matrix
	int *row_offsets_d = NULL;
	int *col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) Copy values from host to device
	CHECK(cudaMemcpy(row_offsets_d, csrMtx->_row_offsets, (numRowsA+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx->_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx->_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3)Create cuSPARSE handle and descriptors
	// cusparseHandle_t cusparseHandler;
	// cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnVecDescr_t vecX, vecY;

	CHECK_CUSPARSE(cusparseCreateCsr(&mtxA, numRowsA, numColsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, numColsA, dnsVecX_d, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, numRowsA, dnsVecY_d, CUDA_R_64F));


	//(4) Calculate buffer size of SpMV operation
	size_t bufferSize = 0;
	void *dBuffer = NULL;
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5) Perform sparse-dense vector Multiplication
	CHECK_CUSPARSE(cusparseSpMV(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\n~~dnsVecY~~\n");
		print_vector(dnsVecY_d, numRowsA);
	}

	//(6) Free memory and destroy descriptors
	CHECK_CUSPARSE(cusparseDestroySpMat(mtxA));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
	// CHECK_CUSPARSE(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));

} // end of multiply_Sprc_Den_vec


//Input: double *dnsMtxB_d, const CSRMatrix *csrMtx, double *dnsMtxX_d, int numClmB, double * dnxMtxC_d
//Process: perform C = C - AX
//Output: dnsMtxC_d, dense matrix C in device
void den_mtx_subtract_multiply_Sprc_Den_mtx(cusparseHandle_t cusparseHandler, const CSRMatrix *csrMtx, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d) {
	int numRowsA = csrMtx->_numOfRows;
	int numClmsA = csrMtx->_numOfClms;
	int nnz = csrMtx->_numOfnz;

	double alpha = -1.0f;
	double beta = 1.0f;

	bool debug = false;


	//(1) Allocate device memoery for CSR matrix
	int	*row_offsets_d = NULL;
	int *col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) Copy values from host to device
	CHECK(cudaMemcpy(row_offsets_d, csrMtx->_row_offsets, (numRowsA+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx->_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx->_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3) Crate cuSPARSE handle and descriptors
	// cusparseHandle_t cusparseHandler;
	// cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnMatDescr_t mtxB, mtxC;

	CHECK_CUSPARSE(cusparseCreateCsr(&mtxA, numRowsA, numClmsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnMat(&mtxB, numClmsA, numClmsB, numClmsA, dnsMtxX_d, CUDA_R_64F, CUSPARSE_ORDER_COL));
	CHECK_CUSPARSE(cusparseCreateDnMat(&mtxC, numRowsA, numClmsB, numRowsA, dnsMtxC_d, CUDA_R_64F, CUSPARSE_ORDER_COL));

	//(4) Calculate buffer size of Spase by dense matrix mulply operation
    size_t bufferSize = 0;
    void *dBuffer = NULL;
	CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5)Perform sparse-dense matrix Multiplication
	CHECK_CUSPARSE(cusparseSpMM(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\n~~mtxC after cusparseSpMM~~\n\n");
		print_mtx_clm_d(dnsMtxC_d, numRowsA, numClmsB);
	}

	//(6) Free memeory and destroy descriptors
	CHECK_CUSPARSE(cusparseDestroySpMat(mtxA));
	CHECK_CUSPARSE(cusparseDestroyDnMat(mtxB));
	CHECK_CUSPARSE(cusparseDestroyDnMat(mtxC));
	// CHECK_CUSPARSE(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));
} // end ofden_mtx_subtract_multiply_Sprc_Den_mtx



//Input:
//Process: perform vector y = y  - Ax
//Output: dnsVecY_d, dense vector C in device
void den_vec_subtract_multiplly_Sprc_Den_vec(cusparseHandle_t cusparseHandler, const CSRMatrix *csrMtx, double *dnsVecX_d, double *dnsVecY_d){
	int numRowsA = csrMtx->_numOfRows;
	int numColsA = csrMtx->_numOfClms;
	int nnz = csrMtx->_numOfnz;

	double alpha = -1.0f;
	double beta = 1.0f;

	bool debug = false;

	//(1) Allocate device memory for CSR matrix
	int *row_offsets_d = NULL;
	int * col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) copy values from host to devise
	CHECK(cudaMemcpy(row_offsets_d, csrMtx->_row_offsets, (numRowsA + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx->_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx->_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3) Create cuSPARSE handle and descriptors
	// cusparseHandle_t cusparseHandler;
	// cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnVecDescr_t vecX, vecY;

	CHECK_CUSPARSE(cusparseCreateCsr(&mtxA, numRowsA, numColsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, numRowsA, dnsVecX_d, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, numColsA, dnsVecY_d, CUDA_R_64F));

	//(4) Calculate buffer size of SpMV operation
	size_t bufferSize = 0;
	void *dBuffer = NULL;
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5) Perform sparse matrix-vector multiplication
	CHECK_CUSPARSE(cusparseSpMV(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\nVecY = vecY - mtxA * vecX with sparse function");
		printf("\n~~dnsVecY~~\n");
		print_vector(dnsVecY_d, numRowsA);
	}

	//(6) Free memory and 
	CHECK_CUSPARSE(cusparseDestroySpMat(mtxA));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
	// CHECK_CUSPARSE(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));


} // end of den_mtx_subtract_multiplly_Sprc_Den_vec


double validateCG(const CSRMatrix *csrMtx, int numOfA, double *dnsVecX_d, double* dnsVecY_d)
{
	double residual = 0.0f;

	cublasHandle_t cublasHandler = NULL;
	cusparseHandle_t cusparseHandler = NULL;
	CHECK_CUSPARSE(cusparseCreate(&cusparseHandler));
	CHECK_CUBLAS(cublasCreate(&cublasHandler));
	
	den_vec_subtract_multiplly_Sprc_Den_vec(cusparseHandler, csrMtx, dnsVecX_d, dnsVecY_d);
	CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, dnsVecY_d, 1, dnsVecY_d, 1, &residual));	

	CHECK_CUBLAS(cublasDestroy(cublasHandler));

	return sqrt(residual);

}

//Input: CSRMatrix *csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d
//Process: perform R = B - AX, then calculate the first column vector 2 norms
//Output: double twoNorms
double validateBFBCG(const CSRMatrix *csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d)
{
	bool debug = false;
	
	cublasHandle_t cublasHandler = NULL;
	cusparseHandle_t cusparseHandler = NULL;
	CHECK_CUSPARSE(cusparseCreate(&cusparseHandler));
	CHECK_CUBLAS(cublasCreate(&cublasHandler));

	den_mtx_subtract_multiply_Sprc_Den_mtx(cusparseHandler, csrMtx, dnsMtxX_d, numClmsB, dnsMtxC_d);
	if(debug){
		printf("\n\nmtxR = B - AX\n");
		printf("~~mtxR~~\n\n");
		print_mtx_clm_d(dnsMtxC_d, numOfA, numClmsB);
	}
	


	double twoNorms = 0.0f;
	calculateResidual(cublasHandler, dnsMtxC_d, numOfA, numClmsB, twoNorms);

	CHECK_CUBLAS(cublasDestroy(cublasHandler));

	return twoNorms;
}





#endif // CUSPARSE_UTIL_H