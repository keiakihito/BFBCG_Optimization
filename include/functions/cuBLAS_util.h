#ifndef CUBLAS_UTIL_H
#define CUBLAS_UTIL_H

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "helper.h"



//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix A and matrix B
//Result: matrix C as a result
void multiply_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA);


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix -A and matrix B
//Result: matrix C as a result
void multiply_ngt_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA);


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix C <-  matrix A * matrix B + matrixC with overwrite
//Result: matrix C as a result
void multiply_sum_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA);

//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int number of Row, int number of column
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxC_d, int numOfRow, int numOfClm);

//Input matrix should be column major, overload function
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int number of Row A, int number of column A, int number of cloumn B
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB);

//Input: cublasHandle_t cublasHandler, double* mtxB_d, double* mtxA_d, double* mtxSolX_d, int numOfRowA, int numOfClmB
//Process: Perform R = 
//Output: double* mtxB_d as a result with overwritten
void subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB);


//Input: cublasHandler_t cublasHandler, double* matrix X, int number of row, int number of column
//Process: the function allocate new memory space and tranpose the mtarix X
//Output: double* matrix X transpose
double* transpose_Den_Mtx(cublasHandle_t cublasHandler, double* mtxX_d, int numOfRow, int numOfClm);


//Input: cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double* residual as answer
//Process: Extrace the first column vector from the Residual mtrix and calculate dot product
//Output: double& rsdl_h
void calculateResidual(cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double& rsdl_h);

//Input: cublasHandle_t cublasHandler, double* matrix Residual, int number of row, int number of column
//Process: Extracts the first column vector from the residual matrix,
// 			Calculate dot product of the first column vector, then compare sqare root of dot product with Threashold
//Output: boolean
bool checkStop(cublasHandle_t cublasHandler, double *mtxR_d, int numOfRow, int numOfClm, double const threshold);







//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix A and matrix B
//Result: matrix C as a result
void multiply_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA)
{	
	const double alpha = 1.0f;
	const double beta = 0.0f;

	CHECK_CUBLAS(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfColB, numOfColA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfColA, &beta, mtxC_d, numOfRowA));

}


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix -A and matrix B
//Result: matrix C as a result
void multiply_ngt_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA)
{	
	const double alpha = -1.0f;
	const double beta = 0.0f;

	CHECK_CUBLAS(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfColB, numOfColA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfColA, &beta, mtxC_d, numOfRowA));
}


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix C <-  matrix A * matrix B + matrixC with overwrite
//Result: matrix C as a result
void multiply_sum_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA)
{	
	const double alpha = 1.0f;
	const double beta = 1.0f;

	CHECK_CUBLAS(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfColB, numOfColA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfColA, &beta, mtxC_d, numOfRowA));
}



//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* result matrix C in device, int number of Row, int number of column
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxC_d, int numOfRow, int numOfClm)
{	
	const double alpha = 1.0f;
	const double beta = 0.0f;
	CHECK_CUBLAS(cublasDgemm(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfClm, numOfClm, numOfRow, &alpha, mtxA_d, numOfRow, mtxA_d, numOfRow, &beta, mtxC_d, numOfClm));
	
	// It will be no longer need inside orth(*), and later iteration
	CHECK(cudaFree(mtxA_d)); 
} // end of multiply_Den_ClmM_mtxT_mtx



//Input matrix should be column major, overload function
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int number of Row A, int number of column A, int number of cloumn B
//Process: matrix multiplication matrix A' * matrix B
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB)
{
	
	const double alpha = 1.0f;
	const double beta = 0.0f;

	/*
	Note
	3rd parameter, m: Number of rows of C (or AT), which is numOfColA (K).
	4th parameter, n: Number of columns of C (or B), which is numOfColB (N).
	5th parameter, k: Number of columns of AT (or rows of B), which is numOfRowA (M).
	
	Summary,
	Thinking about 3rd and 4th parameter as matrix C would be colmnA * columnB because A is transposed.
	Then, the 5th parameter is inbetween number as rowA or rowB becuase matrix A is transpose
	*/
	CHECK_CUBLAS(cublasDgemm(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfClmA, numOfClmB, numOfRowA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfRowA, &beta, mtxC_d, numOfClmA));

} // end of multiply_Den_ClmM_mtxT_mtx


//Input: cublasHandle_t cublasHandler, double* mtxB_d, double* mtxA_d, double* mtxSolX_d, int numOfRowA, int numOfClmB
//Process: Perform R = 
//Output: double* mtxB_d as a result with overwritten
void subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB)
{
	const double alpha = -1.0f;
	const double beta = 1.0f;

	CHECK_CUBLAS(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfClmB, numOfClmA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfClmA, &beta, mtxC_d, numOfRowA));
}




//TODO check this function is need or not.
//Input: cublasHandler_t cublasHandler, double* matrix X, int number of row, int number of column
//Process: the function allocate new memory space and tranpose the mtarix X
//Output: double* matrix X transpose
double* transpose_Den_Mtx(cublasHandle_t cublasHandler, double* mtxX_d, int numOfRow, int numOfClm)
{	
	double* mtxXT_d = NULL;
	const double alpha = 1.0f;
	const double beta = 0.0f;

	//Allocate a new memory space for mtxXT
	CHECK(cudaMalloc((void**)&mtxXT_d, numOfRow * numOfClm * sizeof(double)));
	
	//Transpose mtxX
	// CHECK_CUBLAS(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));
    CHECK_CUBLAS(cublasDgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfRow, numOfClm, &alpha, mtxX_d, numOfClm, &beta, mtxX_d, numOfRow, mtxXT_d, numOfRow));

	//Free memory the original matrix X
	CHECK(cudaFree(mtxX_d));

	return mtxXT_d;
}






//Stop condition
//Input: cublasHandle_t cublasHandler, double* matrix Residual, int number of row, int number of column
//Process: Extracts the first column vector from the residual matrix,
// 			Calculate dot product of the first column vector, then compare sqare root of dot product with Threashold
bool checkStop(cublasHandle_t cublasHandler, double *mtxR_d, int numOfRow, int numOfClm, double const threshold)
{
	double *r1_d = NULL;
	double dotPrdct = 0.0f;
	bool debug =false;

	//Extract first column
	CHECK(cudaMalloc((void**)&r1_d, numOfRow * sizeof(double)));
	CHECK(cudaMemcpy(r1_d, mtxR_d, numOfRow * sizeof(double), cudaMemcpyDeviceToDevice));

	if(debug){
		printf("\n\nvector r_1: \n");
		print_vector(r1_d, numOfRow);
	}
	
	//Dot product of r_{1}' * r_{1}, cublasSdot
	CHECK_CUBLAS(cublasDdot(cublasHandler, numOfRow, r1_d, 1, r1_d, 1, &dotPrdct));

	//Square root(dotPrdct)
	if(debug){
		printf("\n\ndot product of r_1: %.10f\n", dotPrdct);
		printf("\n\nsqrt(dot product of r_1): %.10f\n", sqrt(dotPrdct));
		printf("\n\nTHRESHOLD : %f\n", threshold);
	}

	CHECK(cudaFree(r1_d));

	return (sqrt(dotPrdct)< threshold);
}


//Input: cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double* residual as answer
//Process: Extrace the first column vector from the Residual mtrix and calculate dot product
//Output: double& rsdl_h
void calculateResidual(cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double& rsdl_h)
{	
	double* r1_d = NULL;
	bool debug =false;

	//Extract first column
	CHECK(cudaMalloc((void**)&r1_d, numOfRow * sizeof(double)));
	CHECK(cudaMemcpy(r1_d, mtxR_d, numOfRow * sizeof(double), cudaMemcpyDeviceToDevice));

	if(debug){
		printf("\n\nvector r_1: \n");
		print_vector(r1_d, numOfRow);
	}
	
	//Dot product of r_{1}' * r_{1}, cublasSdot
	CHECK_CUBLAS(cublasDdot(cublasHandler, numOfRow, r1_d, 1, r1_d, 1, &rsdl_h));

	//Square root(dotPrdct)
	if(debug){
		printf("\n\ndot product of r_1: %.10f\n", rsdl_h);
		printf("\n\nsqrt(dot product of r_1): %.10f\n", sqrt(rsdl_h));
	}

	// Set residual swuare root of dot product
	rsdl_h = sqrt(rsdl_h);

	CHECK(cudaFree(r1_d));
}






#endif // CUBLAS_UTIL_H