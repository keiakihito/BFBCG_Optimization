#ifndef CUSOLVER_UTIL_H
#define CUSOLVER_UTIL_H

#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "helper.h"


//Inverse funtion with sub functions
//Input: double* matrix A, double* matrix A inverse, int N, number of Row or Column
//Process: Inverse matrix
//Output: mtxA_inv_d;
void inverse_QR_Den_Mtx(cusolverDnHandle_t cusolverHandler, cublasHandle_t cublasHandler,  double* mtxA_d, double* mtxA_inv_d, int N);

//Inverse funtion with sub functions
//Input: double* matrix A, double* matrix A inverse, int N, number of Row or Column
//Process: Inverse matrix
//Output: lfoat mtxQPT_d
void inverse_Den_Mtx(cusolverDnHandle_t cusolverHandler, double* mtxA_d, double* mtxA_inv_d, int N);


//Input: double* identity matrix, int numer of Row, Column
//Process: Creating identity matrix with number of N
//Output: double* mtxI
__global__ void identity_matrix(double* mtxI_d, int N);

//Input: double* identity matrix, int N, which is numer of Row or Column
//Process: Call the kernel to create identity matrix with number of N
//Output: double* mtxI
void createIdentityMtx(double* mtxI_d, int N);

//Input: double* matrix A, int number of Row, int number Of Column
//Process: Compute condition number and check whther it is ill-conditioned or not.
//Output: double condition number
double computeConditionNumber(double* mtxA_d, int numOfRow, int numOfClm);

//Input: cusolverDnHandle_t cusolverHandler, int number of row, int number of column, int leading dimension, double* matrix A
//Process: Extract eigenvalues with full SVD
//Output: double* sngVals_d, singular values in device in column vector
double* extractSngVals(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, double* mtxA_d);







//Inverse with QR decompostion
void inverse_QR_Den_Mtx(cusolverDnHandle_t cusolverHandler, cublasHandle_t cublasHandler,  double* mtxA_d, double* mtxA_inv_d, int N)
{	
	//Check matrix is invertible or not.
	const double CONDITION_NUM_THRESHOLD = 1000;
	double conditionNum = computeConditionNumber(mtxA_d, N, N);
	assert (conditionNum < CONDITION_NUM_THRESHOLD && "\n\n!!ill-conditioned matrix A in inverse function!!\n\n");
	
	bool debug = false;

	double *mtxA_cpy_d = NULL;
	double *tau_d = NULL;
	const int lda = N; //Leading dimention of A


	double *work_d = NULL;
	int *devInfo = NULL;
	int lwork = 0;

	if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }

	//(1) Allocate memoery
	CHECK(cudaMalloc((void**)&mtxA_cpy_d, N * N * sizeof(double)));
	CHECK(cudaMalloc((void**)&tau_d, N * sizeof(double)));
	CHECK(cudaMalloc((void**)&devInfo, N * sizeof(int)));
	
	//(2) Make copy of mtxA
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, N * N * sizeof(double), cudaMemcpyDeviceToDevice));

	if(debug){
        printf("\n\n~~mtxA_cpy_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }
	
	//(3) Create Identity matrix
	createIdentityMtx(mtxA_inv_d, N);
	if(debug){
		printf("\n\n~~mtxA_inv_d (mtxI)~~\n\n");
		print_mtx_clm_d(mtxA_inv_d, N, N);
	}

	//(4)Calculate work space for cusolver
	CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverHandler, N, N, mtxA_cpy_d, N, &lwork));
	CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(double)));
	
	//(5)Perform QR decomposition
	CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverHandler, N, N, mtxA_cpy_d, lda, tau_d, work_d, lwork, devInfo));

	if(debug){
        printf("\n\nAfter QR factorization\n");
        printf("\n\n~~mtxA_cpy_d~~\n");
        print_mtx_clm_d(mtxA_cpy_d, N, N);
    }

	//(6) Solve system
	//Let (QR) * X = I, recall Q * Q^{-1} = Q * Q' = I where Q' is stored to mtxA_inv_d
	//cusolverDnDormqr obtains Q' performing R * X = Q' * I where R is upper triangular matrix
	CHECK_CUSOLVER(cusolverDnDormqr(cusolverHandler, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, N, N, N, mtxA_cpy_d, lda, tau_d, mtxA_inv_d, lda, work_d, lwork, devInfo));
	CHECK(cudaDeviceSynchronize());

	//Solve RX = Q'
	//cublasDtrsm is good for solving triangular linear system
	//The result will be sotred to mtxA_inv_d  
	const double alpha = 1.0;
	CHECK_CUBLAS(cublasDtrsm(cublasHandler, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N, &alpha, mtxA_cpy_d, lda, mtxA_inv_d, lda));
	
	// //Check solver after QR decomposition was successful or not.
	// CHECK(cudaMemcpy(&devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost));


	//Check the result
    if(debug){
        printf("\n\nAfter RX = Q'\n");
        printf("\n\n~~mtxA inverse~~\n\n");
        print_mtx_clm_d(mtxA_inv_d, N, N);
    }

	//(7)Free memoery
	CHECK(cudaFree(mtxA_cpy_d));
	CHECK(cudaFree(tau_d));
	CHECK(cudaFree(work_d));
    CHECK(cudaFree(devInfo));
    
} // end of inverse_QR_Den_Mtx



//Input: double* matrix A, double* matrix A inverse, int N, number of Row or Column
//Process: Inverse matrix
//Output: lfoat mtxQPT_d
void inverse_Den_Mtx(cusolverDnHandle_t cusolverHandler, double* mtxA_d, double* mtxA_inv_d, int N)
{
	double* mtxA_cpy_d = NULL;

	double *work_d = nullptr;

    //The devInfo pointer holds the status information after the LU decomposition or solve operations.
    int *devInfo = nullptr;
    
    /*
    A pivots_d pointer holds the pivot indices generated by the LU decomposition. 
    These indices indicate how the rows of the matrix were permuted during the factorization.
    */
    int *pivots_d = nullptr;
    
    //Status information specific to the LAPACK operations performed by cuSolver.
    // int *lapackInfo = nullptr;

    // Size of the workng space
    int lwork = 0;
	bool debug = false;


    if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }

	//(1) Make copy of mtxA
	CHECK(cudaMalloc((void**)&mtxA_cpy_d, N * N * sizeof(double)));
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, N * N * sizeof(double), cudaMemcpyDeviceToDevice));
	
	if(debug){
		printf("\n\n~~mtxA_cpy_d~~\n\n");
		print_mtx_clm_d(mtxA_cpy_d, N, N);
	}

	//(2) Create Identity matrix
	createIdentityMtx(mtxA_inv_d, N);
	if(debug){
		printf("\n\n~~mtxI~~\n\n");
		print_mtx_clm_d(mtxA_inv_d, N, N);
	}

	//(3)Calculate work space for cusolver
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cusolverHandler, N, N, mtxA_cpy_d, N, &lwork));
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(double)));
	CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    CHECK(cudaMalloc((void**)&pivots_d, N * sizeof(int)));

	//(4.1) Perform the LU decomposition, 
    CHECK_CUSOLVER(cusolverDnDgetrf(cusolverHandler, N, N, mtxA_cpy_d, N, work_d, pivots_d, devInfo));
    cudaDeviceSynchronize();

	//Check LU decomposition was successful or not.
	//If not, it can be ill-conditioned or singular.
	int devInfo_h;
	CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(devInfo_h != 0){
		printf("\n\nLU decomposition failed in the inverse_Den_Mtx, info = %d\n", devInfo_h);
		if(devInfo_h == 11){
			printf("\n!!!The matrix potentially is ill-conditioned or singular!!!\n\n");
			double* mtxA_check_d = NULL;
			CHECK(cudaMalloc((void**)&mtxA_check_d, N * N * sizeof(double)));
			CHECK(cudaMemcpy(mtxA_check_d, mtxA_d, N * N * sizeof(double), cudaMemcpyDeviceToDevice));
			double conditionNum = computeConditionNumber(mtxA_check_d, N, N);
			printf("\n\n🔍Condition number = %f🔍\n\n", conditionNum);
			CHECK(cudaFree(mtxA_check_d));
		}
		exit(1);
	}


    /*
    mtxA_d will be a compact form such that 

    A = LU = | 4  1 |
             | 1  3 |
    
    L = |1    0 |  U = |4    1  |
        |0.25 1 |      |0   2.75|
    
    mtxA_d compact form = | 4      1  |
                          | 0.25  2.75|
    */
	if(debug){
        printf("\n\nAfter LU factorization\n");
        printf("\n\n~~mtxA_cpy_d~~\n");
        print_mtx_clm_d(mtxA_cpy_d, N, N);
    }


    //(4.2)Solve for the iverse, UX = Y
    /*
    A = LU
    A * X = LU * X = I
    L * (UX) = L * Y = I
    UX = Y

	cusolverDnDgetrf is LU decompostion such that A = L * U
	cusolverDnDgetrs is solver such that 
	AX = LUX = L(UX) = LY = I
	So We do LY = I, then UX = Y to solve X
    */

   //Solve UX = I 
    CHECK_CUSOLVER(cusolverDnDgetrs(cusolverHandler, CUBLAS_OP_N, N, N, mtxA_cpy_d, N, pivots_d, mtxA_inv_d, N, devInfo));
	CHECK(cudaDeviceSynchronize());

	//Check solver after LU decomposition was successful or not.
	CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(devInfo_h != 0){
		printf("Solve after LU failed, info = %d\n", devInfo_h);
		exit(1);
	}


	//(5)Free memoery
	CHECK(cudaFree(mtxA_cpy_d));
	CHECK(cudaFree(work_d));
    CHECK(cudaFree(devInfo));
    CHECK(cudaFree(pivots_d));

} // end of inverse_Den_Mtx


//Input: double* identity matrix, int numer of Row, Column
//Process: Creating dense identity matrix with number of N
//Output: double* mtxI
__global__ void identity_matrix(double* mtxI_d, int N)
{	
	//Get global index 
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	//Set boundry condition
	if(idx < (N * N)){
		int glbRow = idx / N;
		int glbClm = idx % N;

		// Index points to the diagonal element
		if(glbRow == glbClm){
			mtxI_d[idx] = 1.0f;
		}else{
			mtxI_d[idx] = 0.0f;
		}// end of if, store diagonal element

	} // end of if, boundtry condition

}// end of identity_matrix


//Input: double* identity matrix, int N, which is numer of Row or Column
//Process: Call the kernel to create identity matrix with number of N
//Output: double* mtxI
void createIdentityMtx(double* mtxI_d, int N)
{		
	// Use a 1D block and grid configuration
    int blockSize = 1024; // Number of threads per block
    int gridSize = ceil((double)N * N / blockSize); // Number of blocks needed

    identity_matrix<<<gridSize, blockSize>>>(mtxI_d, N);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
	}
        
	cudaDeviceSynchronize(); // Ensure the kernel execution completes before proceeding
}



//Input: double* matrix A, int number of Row, int number Of Column
//Process: Compute condition number and check whther it is ill-conditioned or not.
//Output: double condition number
double computeConditionNumber(double* mtxA_d, int numOfRow, int numOfClm)
{
	bool debug = false;

	//Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandler));

	double* sngVals_d = extractSngVals(cusolverHandler, numOfRow, numOfClm, numOfRow, mtxA_d);
	if(debug){
		printf("\n\nsngular values after SVD decomp\n\n");
		print_vector(sngVals_d, numOfClm);
	}

	double* sngVals_h = (double*)malloc(numOfClm * sizeof(double));
	CHECK(cudaMemcpy(sngVals_h, sngVals_d, numOfClm * sizeof(double), cudaMemcpyDeviceToHost));
	double conditionNum = sngVals_h[0] / sngVals_h[numOfClm-1];
	
	CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandler));
	CHECK(cudaFree(sngVals_d));
	free(sngVals_h);
	
	return conditionNum;

} // end of computeConditionNumber



//Input: cusolverDnHandle_t cusolverHandler, int number of row, int number of column, int leading dimension, double* matrix A
//Process: Extract eigenvalues with full SVD
//Output: double* sngVals_d, singular values in device in column vector
double* extractSngVals(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, double* mtxA_d)
{
	
	double *mtxA_cpy_d = NULL; // Need a copy to tranpose mtxZ'

	double *mtxU_d = NULL;
	double *sngVals_d = NULL;
	double *mtxVT_d = NULL;


	/*The devInfo is an integer pointer
    It points to device memory where cuSOLVER can store information 
    about the success or failure of the computation.*/
    int *devInfo = NULL;

    int lwork = 0;//Size of workspace
    //work_d is a pointer to device memory that serves as the workspace for the computation
    //Then passed to the cuSOLVER function performing the computation.
    double *work_d = NULL; // 
    double *rwork_d = NULL; // Place holder
    

    //Specifies options for computing all or part of the matrix U: = ‘A’: 
    //all m columns of U are returned in array
    signed char jobU = 'A';

    //Specifies options for computing all or part of the matrix V**T: = ‘A’: 
    //all N rows of V**T are returned in the array
    signed char jobVT = 'A';

	//Error cheking after performing SVD decomp
	int infoGpu = 0;

	bool debug = false;


	if(debug){
		printf("\n\n~~mtxA~~\n\n");
		print_mtx_clm_d(mtxA_d, numOfRow, numOfClm);
	}


	//(1) Allocate memeory in device
	//Make a copy of mtxZ for mtxZ'
    CHECK(cudaMalloc((void**)&mtxA_cpy_d, numOfRow * numOfClm * sizeof(double)));

	//For SVD decomposition
	CHECK(cudaMalloc((void**)&mtxU_d, numOfRow * numOfClm * sizeof(double)));
	CHECK(cudaMalloc((void**)&sngVals_d, numOfClm * sizeof(double)));
	CHECK(cudaMalloc((void**)&mtxVT_d, numOfClm * numOfClm * sizeof(double)));

	//(2) Copy value to device
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, numOfRow * numOfClm * sizeof(double), cudaMemcpyDeviceToDevice));
	
	
	if(debug){
		printf("\n\n~~mtxA cpy~~\n\n");
		print_mtx_clm_d(mtxA_cpy_d, numOfRow, numOfClm);
	}


	//(4) Calculate workspace for SVD decompositoin
	cusolverDnSgesvd_bufferSize(cusolverHandler, numOfRow, numOfClm, &lwork);
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(double)));
	CHECK((cudaMalloc((void**)&devInfo, sizeof(int))));

    //(3) Compute SVD decomposition
    cusolverDnDgesvd(cusolverHandler, jobU, jobVT, numOfRow, numOfClm, mtxA_cpy_d, ldngDim, sngVals_d, mtxU_d,ldngDim, mtxVT_d, numOfClm, work_d, lwork, rwork_d, devInfo);
	
	//(4) Check SVD decomp was successful. 
	CHECK(cudaMemcpy(&infoGpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(infoGpu != 0){
		printf("\n\n😖😖😖Unsuccessful SVD execution😖😖😖\n");
	}

	// if(debug){
	// 	printf("\n\n~~sngVals_d~~\n\n");
	// 	print_mtx_clm_d(sngVals_d, numOfClm, numOfClm);
	// }

	//(5) Free memoery
	CHECK(cudaFree(work_d));
	CHECK(cudaFree(devInfo));
	CHECK(cudaFree(mtxA_cpy_d));
	CHECK(cudaFree(mtxU_d));
	CHECK(cudaFree(mtxVT_d));


	return sngVals_d;

} // end of extractSngVals





#endif // CUSOLVER_UTIL_H