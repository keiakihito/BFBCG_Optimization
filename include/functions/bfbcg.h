#ifndef BFBCG_H
#define BFBCG_H


#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <sys/time.h>

// helper function CUDA error checking and initialization
#include "helper.h"
#include "cuBLAS_util.h"
#include "cuSPARSE_util.h"
#include "orth_SVD.h"

#include "../struct/CSRMatrix.hpp"




void bfbcg(CSRMatrix *csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX);
void bfbcg_Opt(CSRMatrix *csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX);
double* fillUpVecForBFBCG(double* vecX_h, int numOfA, int numOfColX);



//Breakdown Free Block Conjugate Gradient (BFBCG) function
//Input: double* mtxA_d, double* mtxB_d, double* mtxSolX_d, int numOfA, int numOfColX
//Process: Solve AX = B where A is sparse matrix, X is solutoinn column vectors and B is given column vectors
//Output: double* mtxSolX_d
void bfbcg(CSRMatrix *csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX)
{      
    bool debug = false;
    bool benchmark = true;
    double startTime, endTime; // For bench mark

    int crrntRank = numOfColX;
    //FIXME THRESHOLD 1e-4~ mtxP is too small.
    const double THRESHOLD = 1e-12;
    bool isStop = false;

    double* mtxR_d = NULL; // Residual
    CSRMatrix *csrMtxM = generateSparseIdentityMatrixCSR(numOfA); // Precondtion
    double* mtxZ_d = NULL; // Residual * precondition
    double* mtxP_d = NULL; // Search space
    double* mtxQ_d = NULL; // Q <- A * P
    double* mtxPTQ_d = NULL; // To calculate mtxPTQ_inv_d
    double* mtxPTQ_inv_d = NULL; // Save it for beta calulatoin
    double* mtxPTR_d = NULL; 
    double* mtxAlph_d = NULL; // Alpha
    // double* sclAlph_d = NULL; // Alpha for scaler
    double* alpha_h = NULL; // sclaler for alpha in case alpha is 1 by 1
    double* mtxBta_d = NULL; // Beta
    double* beta_h = NULL; // sclaler for alpha in case alpha is 1 by 1
    double* mtxQTZ_d = NULL; // For calculating mtxBta

    
    //For calculating relative residual during the iteration
    double orgRsdl_h = 0.0f; // Initial residual
    double newRsdl_h = 0.0f; // New residual dring the iteration
    double rltvRsdl_h = 0.0f; // Relateive resitual


    //Crete handler
    cublasHandle_t cublasHandler = NULL;
    cusparseHandle_t cusparseHandler = NULL;
    cusolverDnHandle_t cusolverHandler = NULL;
    
    CHECK_CUBLAS(cublasCreate(&cublasHandler));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandler));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate memory
    CHECK(cudaMalloc((void**)&mtxR_d, numOfA * crrntRank * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxZ_d, numOfA * crrntRank * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxQ_d, numOfA * crrntRank * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxPTQ_d, crrntRank * crrntRank * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxPTQ_inv_d, crrntRank * crrntRank * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxPTR_d, crrntRank * numOfA * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxAlph_d, crrntRank * numOfColX * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxBta_d, crrntRank * numOfColX * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxQTZ_d, crrntRank * numOfA * sizeof(double)));


    alpha_h = (double*)malloc(sizeof(double));
    beta_h = (double*)malloc(sizeof(double));
    

    //(2) Copy memory
    CHECK(cudaMemcpy(mtxR_d, mtxB_d, numOfA * numOfColX * sizeof(double), cudaMemcpyDeviceToDevice));
    if(debug){
        printf("\n\n~~mtxR~~\n\n");
        print_mtx_clm_d(mtxR_d, numOfA, numOfColX);
    }


    //Set up before iterating
    //R <- B - AX
    den_mtx_subtract_multiply_Sprc_Den_mtx(cusparseHandler, csrMtxA, mtxSolX_d, numOfColX, mtxR_d);

    // subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandler, mtxA_d, mtxSolX_d, mtxR_d, numOfA, numOfA, numOfColX);
    
    if(debug){
        printf("\n\n~~mtxR~~\n\n");
        print_mtx_clm_d(mtxR_d, numOfA, numOfColX);
    }

    calculateResidual(cublasHandler, mtxR_d, numOfA, numOfColX, orgRsdl_h);
    if(debug){
        printf("\n\n~~original residual: %f~~\n\n", orgRsdl_h);
    }

    //Z <- MR
    multiply_Sprc_Den_mtx(cusparseHandler, csrMtxM, mtxR_d, numOfColX, mtxZ_d);
    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfA, numOfColX);
        printf("\n\n~~mtxR~~\n\n");
        print_mtx_clm_d(mtxR_d, numOfA, numOfColX);
    }

    //P <- orth(Z), mtxZ will be freed in the function
    orth_SVD(&mtxP_d, mtxZ_d, numOfA, crrntRank, crrntRank);
//    orth_QR(&mtxP_d, mtxZ_d, numOfA, crrntRank, crrntRank);

    if(debug){
        printf("\n\n = =  Current Rank: %d = = \n\n", crrntRank);
        printf("\n\n~~mtxP~~\n\n");
        print_mtx_clm_d(mtxP_d, numOfA, crrntRank);
    }

    //Start iteration
    int counter = 1;
    const int MAX_COUNT = 100;
    while(counter < MAX_COUNT){
        if(debug){
            printf("\n\n\n💫💫💫 Iteration %d 💫💫💫 \n", counter);
            printf("\n= = current Rank: %d = =\n", crrntRank);
        }
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\n\n\n💫💫💫 Iteration %d 💫💫💫 \n", counter);
        }
        
        
        //Q <- AP
        startTime = myCPUTimer();
        multiply_Sprc_Den_mtx(cusparseHandler, csrMtxA, mtxP_d, crrntRank, mtxQ_d);
        endTime = myCPUTimer();
        
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nQ <- AP: %f s \n", endTime - startTime);
        }
        
        if(debug){
            // printf("\n\n~~csrMtxA~~\n\n");
            // print_CSRMtx(csrMtxA);
            printf("\n\n~~mtxQ~~\n\n");
            print_mtx_clm_d(mtxQ_d, numOfA, crrntRank);
        }

        
        //Starting calculate Alpha <- (P'Q)^{-1} * (P'R) with inverse
        //Part (a) (P'Q)
        startTime = myCPUTimer();
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxQ_d, mtxPTQ_d, numOfA, crrntRank, crrntRank);
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nPart(a): (P'Q) %f s \n", endTime - startTime);
        }
        
        if(debug){
            printf("\n\n~~mtxPTQ~~\n\n");
            print_mtx_clm_d(mtxPTQ_d, crrntRank, crrntRank);
        }

        //LU factorization inverse
        // inverse_Den_Mtx(cusolverHandler, mtxPTQ_d, mtxPTQ_inv_d, crrntRank);
        
        
        //Part (b) (P'Q)^{-1}
        //QR decompostion inverse
        //(P'Q)^{-1}, save for the beta calculation
        startTime = myCPUTimer();
        inverse_QR_Den_Mtx(cusolverHandler, cublasHandler, mtxPTQ_d, mtxPTQ_inv_d, crrntRank);
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nPart(b): (P'Q)^{-1} %f s \n", endTime - startTime);
        }
        
        if(debug){
            printf("\n\n~~mtxPTQ_inv~~\n\n");
            print_mtx_clm_d(mtxPTQ_inv_d, crrntRank, crrntRank);
        }

        //Part (C) (P'R)
        startTime = myCPUTimer();
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxR_d, mtxPTR_d, numOfA, crrntRank, numOfColX);
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nPart(C): (P'R) %f s \n", endTime - startTime);
        }
        if(debug){
            printf("\n\n~~mtxPTR~~\n\n");
            print_mtx_clm_d(mtxPTR_d, crrntRank, numOfColX);
        }


        //Alpha <- (P'Q)^{-1} * (P'R)
        //Part (d)
        if(benchmark && crrntRank == 1){
            // Copy data to mtxAlpha to overwrite as an answer
            // CHECK(cudaMalloc((void**)&mtxAlph_d, numOfColX * sizeof(double)));
            CHECK(cudaMemcpy(mtxAlph_d, mtxPTR_d, numOfColX * sizeof(double), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(alpha_h, mtxPTQ_inv_d, sizeof(double), cudaMemcpyDeviceToHost));
            if(debug){
                printf("\n\n = = mtxPTQ_inv_d: %f = = \n", *alpha_h);
            }
            startTime = myCPUTimer();
            CHECK_CUBLAS(cublasDscal(cublasHandler, numOfColX, alpha_h, mtxAlph_d, crrntRank));
            endTime = myCPUTimer();
            if(benchmark && (1 <= counter && counter <= 6)){
                printf("\nPart(d): Alpha <- (P'Q)^{-1} * (P'R): %f s \n", endTime - startTime);
            }
            if(debug){
                printf("\n\n~~mtxAlph_d~~\n\n");
                print_mtx_clm_d(mtxAlph_d, crrntRank, numOfColX);
            }

        }else{
            startTime = myCPUTimer();
            multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxPTQ_inv_d, mtxPTR_d, mtxAlph_d, crrntRank, numOfColX, crrntRank);
            endTime = myCPUTimer();
            if(benchmark && (1 <= counter && counter <= 6)){
                printf("\nPart(d): Alpha <- (P'Q)^{-1} * (P'R): %f s \n", endTime - startTime);
            }
            if(debug){
                printf("\n\n~~mtxAlpha~~\n\n");
                print_mtx_clm_d(mtxAlph_d, crrntRank, numOfColX);
            }
        }


        //X_{i+1} <- x_{i} + P * alpha
        startTime = myCPUTimer();
        multiply_sum_Den_ClmM_mtx_mtx(cublasHandler, mtxP_d, mtxAlph_d, mtxSolX_d, numOfA, numOfColX, crrntRank);
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nX_{i+1} <- x_{i} + P * alpha: %f s \n", endTime - startTime);
        }

        if(debug){
            printf("\n\n~~mtxSolX_d~~\n\n");
            print_mtx_clm_d(mtxSolX_d, numOfA, numOfColX);
        }

        //R_{i+1} <- R_{i} - Q * alpha
        startTime = myCPUTimer();
        subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandler, mtxQ_d, mtxAlph_d, mtxR_d, numOfA, crrntRank, numOfColX);
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nR_{i+1} <- R_{i} - Q * alpha: %f s \n", endTime - startTime);
        }

        if(debug){
            printf("\n\n~~mtxR_d~~\n\n");
            print_mtx_clm_d(mtxR_d, numOfA, numOfColX);
        }
        
        calculateResidual(cublasHandler, mtxR_d, numOfA, numOfColX, newRsdl_h);
        rltvRsdl_h = newRsdl_h / orgRsdl_h; // Calculate Relative Residue
        if(debug){
            printf("\n🫥Relative Residue: %f🫥\n\n", rltvRsdl_h);
        }
        
    

        //If it is converged, then stopped.
        isStop = checkStop(cublasHandler, mtxR_d, numOfA, numOfColX, THRESHOLD);
        if(isStop)
        {
            printf("\n🫥Relative Residue: %f🫥\n\n", rltvRsdl_h);
            // printf("\n\n🌀🌀🌀CONVERGED🌀🌀🌀\n\n");
            break;
        }

        // Z_{i+1} <- MR_{i+1}
        startTime = myCPUTimer();
        multiply_Sprc_Den_mtx(cusparseHandler, csrMtxM, mtxR_d, numOfColX, mtxZ_d);
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nZ_{i+1} <- MR_{i+1}: %f s \n", endTime - startTime);
        }
        if(debug){
            printf("\n\n~~mtxZ~~\n\n");
            print_mtx_clm_d(mtxZ_d, numOfA, numOfColX);
            printf("\n\n~~mtxR~~\n\n");
            print_mtx_clm_d(mtxR_d, numOfA, numOfColX);
        }



        //(Q'Z_{i+1})
        startTime = myCPUTimer();
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_d, mtxZ_d, mtxQTZ_d, numOfA, crrntRank, numOfColX);
        if(debug){
            printf("\n\n~~mtxQTZ~~\n\n");
            print_mtx_clm_d(mtxQTZ_d, crrntRank, numOfColX);
        }

        //beta <- -(P'Q)^{-1} * (Q'Z_{i+1})
        if(benchmark && crrntRank == 1){
            
            // Copy data to mtxBta to overwrite as an answer
            CHECK(cudaMemcpy(mtxBta_d, mtxQTZ_d, numOfColX * sizeof(double), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(beta_h, mtxPTQ_inv_d, crrntRank * sizeof(double), cudaMemcpyDeviceToHost));
            *beta_h *= -1.0f;

            CHECK_CUBLAS(cublasDscal(cublasHandler, numOfColX, beta_h, mtxBta_d, crrntRank));
            endTime = myCPUTimer();
            if(benchmark && (1 <= counter && counter <= 6)){
                printf("\nbeta <- -(P'Q)^{-1} * (Q'Z_{i+1}): %f s \n", endTime - startTime);
            }

            if(debug){
                printf("\n\n~~mtxBta_d~~\n\n");
                print_mtx_clm_d(mtxBta_d, crrntRank, numOfColX);
            }

        }else{
            multiply_ngt_Den_ClmM_mtx_mtx(cublasHandler, mtxPTQ_inv_d, mtxQTZ_d, mtxBta_d, crrntRank, numOfColX, crrntRank);
            endTime = myCPUTimer();
            if(benchmark && (1 <= counter && counter <= 6)){
                printf("\nbeta <- -(P'Q)^{-1} * (Q'Z_{i+1}): %f s \n", endTime - startTime);
            }
            if(debug){
                printf("\n\n~~mtxBta_d~~\n\n");
                print_mtx_clm_d(mtxBta_d, crrntRank, numOfColX);
            }
        }



        //P_{i+1} = orth(Z_{i+1} + p * beta)
        //Z_{i+1} <- Z_{i+1} + p * beta overwrite with Sgemm function
        //Then P_{i+1} <- orth(Z_{ i+1})
        startTime = myCPUTimer();
        multiply_sum_Den_ClmM_mtx_mtx(cublasHandler, mtxP_d, mtxBta_d, mtxZ_d, numOfA, numOfColX, crrntRank);
        
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\n(*) <- Z_{i+1} + p * beta: %f s \n", endTime - startTime);
        }
        if(debug){
            printf("\n\n~~ (mtxZ_{i+1}_d + p * beta) ~~\n\n");
            print_mtx_clm_d(mtxZ_d, numOfA, numOfColX);
        }


        //To update matrix P
        startTime = myCPUTimer();
        orth_SVD(&mtxP_d, mtxZ_d, numOfA, numOfColX, crrntRank);
//        orth_QR(&mtxP_d, mtxZ_d, numOfA, crrntRank, crrntRank);
        endTime = myCPUTimer();
        if(benchmark && (1 <= counter && counter <= 6)){
            printf("\nP_{i+1} = orth(*): %f s \n", endTime - startTime);
        }
        if(debug){
            printf("\n\n~~ mtxP_d <- orth(*)~~\n\n");
            print_mtx_clm_d(mtxP_d, numOfA, crrntRank);
            printf("\n\n= = current Rank: %d = = \n\n", crrntRank);
        }

        if(crrntRank == 0)
        {
            printf("\n\n!!!Current Rank became 0!!!\n 🔸Exit iteration🔸\n");
            printf("\n🫥Relative Residue: %f🫥\n\n", rltvRsdl_h);
            break;
        }
        
        printf("\n\n= = current Rank: %d = = \n\n", crrntRank);

        counter++;
    } // end of while

    if(counter == MAX_COUNT){
        printf("\n\nNOT CONVERGED\n\n");
    }




    //()Free memoery
    CHECK_CUBLAS(cublasDestroy(cublasHandler));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandler));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverHandler));

    CHECK(cudaFree(mtxR_d));
    CHECK(cudaFree(mtxZ_d));
    CHECK(cudaFree(mtxQ_d));
    CHECK(cudaFree(mtxPTQ_d));
    CHECK(cudaFree(mtxPTQ_inv_d));
    CHECK(cudaFree(mtxPTR_d));
    CHECK(cudaFree(mtxQTZ_d));
    free(alpha_h);
    free(beta_h);
    
} // end of bfbcg


//Optimize vertions BFBCG
void bfbcg_Opt(CSRMatrix *csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX){
    
}






double* fillUpVecForBFBCG(double* vecX_h, int numOfA, int numOfColX){
    srand(time(0));

    //Allocate memory for the marix with numOfA rows and numOfColX columns
    double *mtxX_h = new double[numOfA * numOfColX];

    //Copy vecX_h into the first colum of mtxX_h
    for(int i = 0; i < numOfA; i++){
        // Copy vecX_h to the first column of matrix X
        mtxX_h[i] = vecX_h[i]; 
    }

    //Fill the remaining columns of mtxX_h with random values between -1.0 and 1.0
    //i starts fron the second column
    for(int j = 1; j < numOfColX; j++){
        for(int i = 0; i < numOfA; i++){
            mtxX_h[(j*numOfA) + i] = -1.0 + static_cast<double>(rand())/(static_cast<double>(RAND_MAX)/2.0); 
        }// end of i
    } // end of j

    return mtxX_h;
}



#endif // BFBCG_H