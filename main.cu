

// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <sstream>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include <cusolverDn.h>
#include<sys/time.h>

#include "include/functions/helper.h"
#include "include/functions/bfbcg.h"
#include "include/functions/pcg.h"
#include "include/struct/CSRMatrix.hpp"



int main(){
    //Base size
	const char*  mtxFile = "files/198147_by_198147_MatrixA_1000.mm";
	const char*  vecFile = "files/198147_by_1_Vector_b_1000.mm";
    
    //Bigger size
    // const char*  mtxFile = "files/3151875_by_3151875MatrixA.mm";
	// const char*  vecFile = "files/3151875_by_1_Vector_b.mm";
    
    const int NUM_OF_CLM_VEC = 5;
    int size;
    bool optimize = false;
    bool naive = false;

    //Warm up the GPU explicitly.
    warmUpGPU();
    warmUpQR(NUM_OF_CLM_VEC);
	
    // (0) Extract CSRMatrix A, vector b
	CSRMatrix* csrMtxA = importCSRMatrixFromMM(mtxFile);
	// std::cout<<csrMtxA->_row_offsets[size-1]<<std::endl;
	double* vecB_h = importVectorFromMM(vecFile, &size);  // Pass the address of size
	double* vecX_h = copyVector(vecB_h, size);
	const int NUM_OF_A = size; // It is mtxA.size from ClothSimulation project
    

    //(0.1) Use the fillUpVecForBFBCG function to create the matrix for X and B
    double* mtxX_h = fillUpVecForBFBCG(vecX_h, NUM_OF_A, NUM_OF_CLM_VEC);
    double* mtxB_h = fillUpVecForBFBCG(vecB_h, NUM_OF_A, NUM_OF_CLM_VEC);

    //(1) Allocate memory on the GPU
    double* mtxX_d = nullptr;
    double* mtxB_d = nullptr;
    double* vecB_d = nullptr;
    CHECK(cudaMalloc((void**)&mtxX_d, NUM_OF_A * NUM_OF_CLM_VEC * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d, NUM_OF_A * NUM_OF_CLM_VEC * sizeof(double)));
    CHECK(cudaMalloc((void**)&vecB_d, NUM_OF_A * sizeof(double)));
    
     //(2) Copy Data from CPU to GPU
    CHECK(cudaMemcpy(mtxX_d, mtxX_h, NUM_OF_A * NUM_OF_CLM_VEC * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, NUM_OF_A * NUM_OF_CLM_VEC * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vecB_d, vecB_h, NUM_OF_A * sizeof(double), cudaMemcpyHostToDevice));
    if(optimize){
        std::cout << "\n~~Optimized bfbcg test~~\n\n";
    }else if(naive){
        //(3) Call bfbcg to utilize CUDA functions
        std::cout << "\n~~Naive bfbcg test~~";
        bfbcg(csrMtxA, mtxX_d, mtxB_d, NUM_OF_A, NUM_OF_CLM_VEC);
    }else{
        std::cout << "\n~~pcg test~~";
        pcg(csrMtxA, mtxX_d, vecB_d, NUM_OF_A);
    }
    

  
    //(4) Free GPU and CPU memory
    CHECK(cudaFree(mtxX_d));
    CHECK(cudaFree(mtxB_d));
    CHECK(cudaFree(vecB_d));
    delete[] mtxX_h;
    delete[] mtxB_h;
    delete[] vecX_h;
    delete[] vecB_h;
    delete csrMtxA;

    std::cout << "\n\n✅✅BFBCG test done successfully✅✅\n\n";
	return 0;
}


