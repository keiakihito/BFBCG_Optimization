#ifndef ORTH_SVD_H
#define ORTH_SVD_H


#include <iostream>
#include <cuda_runtime.h>
// #include <cusparse.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

// helper function CUDA error checking and initialization
#include "helper.h"
#include "cuBLAS_util.h"
#include "cuSOLVER_util.h"



//Input: double* mtxY_hat_d, double* mtxZ, int number of Row, int number Of column, int & currentRank
//Process: the function extracts orthonormal set from the matrix Z
//Output: double* mtxY_hat, the orthonormal set of matrix Z.
void orth_SVD(double** mtxY_hat_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank);


//For orth sub functions
//Input: cusolverDnHandler, int number of row, int number of column, int leading dimensino, 
//		 double* matrix A, double* matrix U, double* vector singlluar values, double* matrix V tranpose
//Process: Singluar Value Decomposion
//Output: double* Matrix U, double* singular value vectors, double* Matrix V transpose
void SVD_Decmp(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, double* mtxA_d, double* mtxU_d, double* sngVals_d, double*mtxVT_d);

//Input: double* matrix V, int number of Row and Column, int current rank
//Process: the functions truncates the matrix V with current rank
//Output: double* matrix V truncated.
double* truncate_Den_Mtx(double* mtxV_d, int numOfN, int currentRank);

//Input: singluar values, int currnet rank, double threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(double* sngVals_d, int currentRank, double threashold);


//Input: double* mtxY, product of matrix Z * matrix U, int number of row, int number of column 
//Process: the function calls kernel and normalize each column vector 
//Output: double* mtxY_d, which will be updated as normalized matrix Y hat.
void normalize_Den_Mtx(cublasHandle_t cublasHandler, double* mtxY_d, int numOfRow, int numOfCol);



//Orth functions
//Input: double* mtxZ, int number of Row, int number Of column, int & currentRank
//Process: the function extracts orthonormal set from the matrix Z
//Output: double* mtxY_hat, the orthonormal set of matrix Z.
void orth_SVD(double** mtxY_hat_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank)
{	
	bool benchmark = true;
	double startTime, endTime; // For bench mark
	/*
	Pseudocode
	// Mayby need to make a copy of mtxZ
	Transpose Z
	Multiply mtxS <- mtxZ' * mtxZ
	Perform SVD
	Transpoze mtxVT, and get mtxV
	Call set Rank
	if(newRank < currentRank){
		Trancate mtxV
		currentRank <- newRank
	}else{
		continue;
	}
	Mutiply mtxY <- mtxZ * mtxV 
	Normalize mtxY <- mtxY
	Return mtxY
	*/

	double *mtxY_d = NULL; // Orthonormal set, serach diretion
	double *mtxZ_cpy_d = NULL; // Need a copy to tranpose mtxZ'
	double *mtxS_d = NULL;

	double *mtxU_d = NULL;
	double *sngVals_d = NULL;
	double *mtxV_d = NULL;
	double *mtxVT_d = NULL;
	double *mtxV_trnc_d = NULL;

	const double THREASHOLD = 1e-12;

	bool debug = false;
	

	if(debug){
		printf("\n\n~~mtxZ~~\n\n");
		print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
	}


	//(1) Allocate memeory in device
	//Make a copy of mtxZ for mtxZ'
	startTime = myCPUTimer();
    CHECK(cudaMalloc((void**)&mtxZ_cpy_d, numOfRow * numOfClm * sizeof(double)));
	CHECK(cudaMalloc((void**)&mtxS_d, numOfRow * numOfClm * sizeof(double)));
	
	//For SVD decomposition
	CHECK(cudaMalloc((void**)&mtxU_d, numOfRow * numOfClm * sizeof(double)));
	CHECK(cudaMalloc((void**)&sngVals_d, numOfClm * sizeof(double)));
	CHECK(cudaMalloc((void**)&mtxVT_d, numOfClm * numOfClm * sizeof(double)));

	//(2) Copy value to device
	CHECK(cudaMemcpy(mtxZ_cpy_d, mtxZ_d, numOfRow * numOfClm * sizeof(double), cudaMemcpyDeviceToDevice));
	
	
	if(debug){
		printf("\n\n~~mtxZ cpy~~\n\n");
		print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
	}

	//(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandler));
    CHECK_CUBLAS(cublasCreate(&cublasHandler));
	endTime = myCPUTimer();
	if(benchmark){
		printf("\n\n~~ inside orth(*) ~~\n");
        printf("\nSet up memory and handler %f s \n", endTime - startTime);
    }
	//(4) Perform orthonormal set prodecure
	//(4.1) Mutiply mtxS <- mtxZ' * mtxZ
	//mtxZ_cpy will be free after multiplication
	startTime = myCPUTimer();
	multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxZ_cpy_d, mtxS_d, numOfRow, numOfClm);
	cudaDeviceSynchronize();
	endTime = myCPUTimer();
	if(benchmark){
        printf("\nPart(a): mtxS <- mtxZ' * mtxZ %f s \n", endTime - startTime);
    }
	if(debug){
		printf("\n\n~~mtxS ~~\n\n");
		print_mtx_clm_d(mtxS_d, numOfClm, numOfClm);
	}

	//(4.2)SVD Decomposition
	startTime = myCPUTimer();
	SVD_Decmp(cusolverHandler, numOfClm, numOfClm, numOfClm, mtxS_d, mtxU_d, sngVals_d, mtxVT_d);
	cudaDeviceSynchronize();
	endTime = myCPUTimer();
	if(benchmark){
        printf("\nPart(b): SVD Decomp for mtxS %f s \n", endTime - startTime);
    }
	if(debug){
		printf("\n\n~~mtxU ~~\n\n");
		print_mtx_clm_d(mtxU_d, numOfClm, numOfClm);
		printf("\n\n~~sngVals ~~\n\n");
		print_mtx_clm_d(sngVals_d, numOfClm, 1);
		printf("\n\n~~mtxVT ~~\n\n");
		print_mtx_clm_d(mtxVT_d, numOfClm, numOfClm);
	}	

	//(4.3) Transpose mtxV <- mtxVT'
	//mtxVT_d will be free inside function
	startTime = myCPUTimer();
	mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, numOfClm, numOfClm);
	cudaDeviceSynchronize();
	endTime = myCPUTimer();
	if(benchmark){
    	printf("\nPart(c): Transpose mtxV <- mtxVT' %f s \n", endTime - startTime);
    }
	if(debug){
		printf("\n\n~~mtxV ~~\n\n");
		print_mtx_clm_d(mtxV_d, numOfClm, numOfClm);
	}	

	//(4.4) Set current rank
	startTime = myCPUTimer();
	currentRank = setRank(sngVals_d, currentRank, THREASHOLD);
	endTime = myCPUTimer();
	if(benchmark){
    	printf("\nPart(d): Calculate current rank %f s \n", endTime - startTime);
    }
	if(debug){
		printf("\n\n~~ new rank = %d ~~\n\n", currentRank);
	}

	if(currentRank == 0){
		debug = false;
	}

	//(4.5) Truncate matrix V
	//mtxV_d will be free after truncate_Den_Mtx
	startTime = myCPUTimer();
	mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, numOfClm, currentRank);
	cudaDeviceSynchronize();
	endTime = myCPUTimer();
	if(benchmark){
    	printf("\nPart(e): Truncate matrix V %f s \n", endTime - startTime);
    }
	if(debug){
		printf("\n\n~~mtxV_Trnc ~~\n\n");
		print_mtx_clm_d(mtxV_trnc_d, numOfClm, currentRank);
	}	

	//(4.6) Multiply matrix Y <- matrix Z * matrix V Truncated
	startTime = myCPUTimer();
	CHECK(cudaMalloc((void**)&mtxY_d, numOfRow * currentRank * sizeof(double)));
	multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxZ_d, mtxV_trnc_d, mtxY_d, numOfRow, currentRank, numOfClm);
	cudaDeviceSynchronize();
	endTime = myCPUTimer();
	if(benchmark){
    	printf("\nPart(f): mtxY <- mtxZ * mtxV Truncated %f s \n", endTime - startTime);
    }
	if(debug){
		printf("\n\n~~mtxY ~~\n\n");
		print_mtx_clm_d(mtxY_d, numOfRow, currentRank);
	}

	//(4.7) Normalize matrix Y_hat <- normalize_Den_Mtx(mtxY_d)
	// normalize_Den_Mtx(mtxY_d, numOfRow, currentRank);
	startTime = myCPUTimer();
	normalize_Den_Mtx(cublasHandler, mtxY_d, numOfRow, currentRank);
	cudaDeviceSynchronize();
	endTime = myCPUTimer();
	if(benchmark){
    	printf("\nPart(g): Normalize mtxY %f s \n", endTime - startTime);
		printf("\n~~Exit orth(*)~~ \n\n");
    }
	if(debug){
		printf("\n\n~~mtxY hat <- orth(*) ~~\n\n");
		print_mtx_clm_d(mtxY_d, numOfRow, currentRank);
	}

	//(4.6) Check orthogonality
	if(debug){
		//Check the matrix Y hat column vectors are orthogonal eachother
		double* mtxI_d = NULL;
		double* mtxY_cpy_d = NULL;
		CHECK(cudaMalloc((void**)&mtxI_d, currentRank * currentRank * sizeof(double)));
		CHECK(cudaMalloc((void**)&mtxY_cpy_d, numOfRow * currentRank * sizeof(double)));
	    CHECK(cudaMemcpy(mtxY_cpy_d, mtxY_d, numOfRow * currentRank * sizeof(double), cudaMemcpyDeviceToDevice));

		//After this function mtxY_cpy_d will be free.
		multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxY_cpy_d, mtxI_d, numOfRow, currentRank);
		
		printf("\n\n~~~~Orthogonality Check (should be close to identity matrix)~~\n\n");
		print_mtx_clm_d(mtxI_d, currentRank, currentRank);
		CHECK(cudaFree(mtxI_d));
	}

	//(5)Pass the address to the provided pointer, updating orhtonomal set
	CHECK(cudaFree(*mtxY_hat_d));
	*mtxY_hat_d = NULL;
	*mtxY_hat_d = mtxY_d;

	if(debug){
		printf("\n\n~~mtxY hat <- orth(*) ~~\n\n");
		print_mtx_clm_d(*mtxY_hat_d, numOfRow, currentRank);
	}


	//(6) Free memory
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandler));
    CHECK_CUBLAS(cublasDestroy(cublasHandler));

	CHECK(cudaFree(mtxS_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_trnc_d));
}




//Input: cusolverDnHandler, int number of row, int number of column, int leading dimensino, 
//		 double* matrix A, double* matrix U, double* vector singlluar values, double* matrix V tranpose
//Process: Singluar Value Decomposion
//Output: double* Matrix U, double* singular value vectors, double* Matrix V transpose
void SVD_Decmp(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, double* mtxA_d, double* mtxU_d, double* sngVals_d, double*mtxVT_d)
{	
	//Make a copy of matrix A to aboid value changing though SVD Decomposion
	double* mtxA_cpy_d = NULL;

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
    signed char jobU = 'S';

    //Specifies options for computing all or part of the matrix V**T: = ‘A’: 
    //all N rows of V**T are returned in the array
    signed char jobVT = 'S';

	//Error cheking after performing SVD decomp
	int infoGpu = 0;


	//(1) Allocate memoery, and copy value
	CHECK(cudaMalloc((void**)&mtxA_cpy_d, numOfRow * numOfClm * sizeof(double))); 
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, numOfRow * numOfClm * sizeof(double), cudaMemcpyDeviceToDevice));

	CHECK((cudaMalloc((void**)&devInfo, sizeof(int))));

	//(2) Calculate workspace for SVD decompositoin
	CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(cusolverHandler, numOfRow, numOfClm, &lwork));
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(double)));


    //(3) Compute SVD decomposition
    CHECK_CUSOLVER(cusolverDnDgesvd(cusolverHandler, jobU, jobVT, numOfRow, numOfClm, mtxA_cpy_d, ldngDim, sngVals_d, mtxU_d,ldngDim, mtxVT_d, numOfClm, work_d, lwork, rwork_d, devInfo));
	
	//(4) Check SVD decomp was successful. 
	CHECK(cudaMemcpy(&infoGpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(infoGpu != 0){
		printf("\n\n😖😖😖Unsuccessful SVD execution😖😖😖\n");
	}

	//(5) Free memoery
	CHECK(cudaFree(work_d));
	CHECK(cudaFree(devInfo));
	CHECK(cudaFree(mtxA_cpy_d));

	return;
}

//Input: double* matrix V, int number of Row and Column, int current rank
//Process: the functions truncates the matrix V with current rank
//Output: double* matrix V truncated.
double* truncate_Den_Mtx(double* mtxV_d, int numOfN, int currentRank)
{	
	//Allocate memoery for truncated matrix V
	double* mtxV_trnc_d = NULL;
	CHECK(cudaMalloc((void**)&mtxV_trnc_d, numOfN * currentRank * sizeof(double)));

	//Copy value from the original matrix until valid column vectors
	CHECK(cudaMemcpy(mtxV_trnc_d, mtxV_d, numOfN * currentRank * sizeof(double), cudaMemcpyDeviceToDevice));

	//Make sure memoery Free full matrix V.
	CHECK(cudaFree(mtxV_d));

	//Return truncated matrix V.
	return mtxV_trnc_d;
}



//Input: singluar values, int currnet rank, double threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(double* sngVals_d, int currentRank, double threashold)
{	
	int newRank = 0;

	//Allcoate in heap to copy value from device
	double* sngVals_h = (double*)malloc(currentRank * sizeof(double));
	// Copy singular values from Device to count eigen values
	CHECK(cudaMemcpy(sngVals_h, sngVals_d, currentRank * sizeof(double), cudaMemcpyDeviceToHost));

	for(int wkr = 0; wkr < currentRank; wkr++){
		if(sngVals_h[wkr] > threashold){
			newRank++;
		} // end of if
	} // end of for

	free(sngVals_h);

	return newRank;
}


//Overloadfunction
void normalize_Den_Mtx(cublasHandle_t cublasHandler, double* mtxY_d, int numOfRow, int numOfCol)
{	
	bool debug = false;
	
	//Make an array for scalars each column vector
	double* norms_h = (double*)malloc(numOfCol * sizeof(double));
	
	//Compute the 2 norms for each column vectors
	for (int wkr = 0; wkr < numOfCol; wkr++){
		CHECK_CUBLAS(cublasDnrm2(cublasHandler, numOfRow, mtxY_d + (wkr * numOfRow), 1, &norms_h[wkr]));
	}

	if(debug){
		for(int wkr = 0; wkr < numOfCol; wkr++){
			printf("\ntwoNorm_h %f\n", norms_h[wkr]);
		}
	}

	//Normalize each column vector
	for(int wkr = 0; wkr < numOfCol; wkr++){
		double scalar = 1.0f / norms_h[wkr];
		CHECK_CUBLAS(cublasDscal(cublasHandler, numOfRow, &scalar, mtxY_d + (wkr * numOfRow), 1));
	}

	free(norms_h);

} // end of normalize_Den_Mtx








#endif // ORTH_SVD_H