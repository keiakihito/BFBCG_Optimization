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
//#include "orth_QR.h"



void bfbcg(CSRMatrix &csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX);
void bfbcg_BenchMark(CSRMatrix &csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX, BenchMarkBFBCG& bmBFBCG);

//Breakdown Free Block Conjugate Gradient (BFBCG) function
//Input: double* mtxA_d, double* mtxB_d, double* mtxSolX_d, int numOfA, int numOfColX
//Process: Solve AX = B where A is sparse matrix, X is solutoinn column vectors and B is given column vectors
//Output: double* mtxSolX_d
void bfbcg(CSRMatrix &csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX)
{      
    bool debug = false;
    double startTime, endTime; // For bench mark

    int crrntRank = numOfColX;
    //FIXME THRESHOLD 1e-4~ mtxP is too small.
    const double THRESHOLD = 1e-12;
    bool isStop = false;

    double* mtxR_d = NULL; // Residual
    CSRMatrix csrMtxM = generateSparseIdentityMatrixCSR(numOfA); // Precondtion
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

        printf("\n\n\n💫💫💫 Iteration %d 💫💫💫 \n", counter);
        printf("\n= = current Rank: %d = =\n", crrntRank);
        
        
        //Q <- AP
        startTime = myCPUTimer();
        multiply_Sprc_Den_mtx(cusparseHandler, csrMtxA, mtxP_d, crrntRank, mtxQ_d);
        endTime = myCPUTimer();
        
        if(counter == 1){
            printf("\nQ <- AP: %f s \n", endTime - startTime);
        }
        
        if(debug){
            // printf("\n\n~~csrMtxA~~\n\n");
            // print_CSRMtx(csrMtxA);
            printf("\n\n~~mtxQ~~\n\n");
            print_mtx_clm_d(mtxQ_d, numOfA, crrntRank);
        }

        //(P'Q)^{-1}, save for the beta calculation
        startTime = myCPUTimer();
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxQ_d, mtxPTQ_d, numOfA, crrntRank, crrntRank);
        if(debug){
            printf("\n\n~~mtxPTQ~~\n\n");
            print_mtx_clm_d(mtxPTQ_d, crrntRank, crrntRank);
        }

        //Check matrix is invertible or not.
        const double CONDITION_NUM_THRESHOLD = 1000;
        double conditionNum = computeConditionNumber(mtxPTQ_d, crrntRank, crrntRank);
        assert (conditionNum < CONDITION_NUM_THRESHOLD && "\n\n!!ill-conditioned matrix A in inverse function!!\n\n");
        
        //LU factorization inverse
        // inverse_Den_Mtx(cusolverHandler, mtxPTQ_d, mtxPTQ_inv_d, crrntRank);
        
        //QR decompostion inverse
        inverse_QR_Den_Mtx(cusolverHandler, cublasHandler, mtxPTQ_d, mtxPTQ_inv_d, crrntRank);
        if(debug){
            printf("\n\n~~mtxPTQ_inv~~\n\n");
            print_mtx_clm_d(mtxPTQ_inv_d, crrntRank, crrntRank);
        }

        //(P'R)
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxR_d, mtxPTR_d, numOfA, crrntRank, numOfColX);
        if(debug){
            printf("\n\n~~mtxPTR~~\n\n");
            print_mtx_clm_d(mtxPTR_d, crrntRank, numOfColX);
        }


        //Alpha <- (P'Q)^{-1} * (P'R)
        if(crrntRank == 1){
            // Copy data to mtxAlpha to overwrite as an answer
            // CHECK(cudaMalloc((void**)&mtxAlph_d, numOfColX * sizeof(double)));
            CHECK(cudaMemcpy(mtxAlph_d, mtxPTR_d, numOfColX * sizeof(double), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(alpha_h, mtxPTQ_inv_d, sizeof(double), cudaMemcpyDeviceToHost));
            if(debug){
                printf("\n\n = = mtxPTQ_inv_d: %f = = \n", *alpha_h);
            }
            
            CHECK_CUBLAS(cublasDscal(cublasHandler, numOfColX, alpha_h, mtxAlph_d, crrntRank));
            endTime = myCPUTimer();
            if(counter == 1){
                printf("\nAlpha <- (P'Q)^{-1} * (P'R): %f s \n", endTime - startTime);
            }
            if(debug){
                printf("\n\n~~mtxAlph_d~~\n\n");
                print_mtx_clm_d(mtxAlph_d, crrntRank, numOfColX);
            }

        }else{
            multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxPTQ_inv_d, mtxPTR_d, mtxAlph_d, crrntRank, numOfColX, crrntRank);
            endTime = myCPUTimer();
            if(counter == 1){
                printf("\nAlpha <- (P'Q)^{-1} * (P'R): %f s \n", endTime - startTime);
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
        if(counter == 1){
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
        if(counter == 1){
            printf("\nR_{i+1} <- R_{i} - Q * alpha: %f s \n", endTime - startTime);
        }

        if(debug){
            printf("\n\n~~mtxR_d~~\n\n");
            print_mtx_clm_d(mtxR_d, numOfA, numOfColX);
        }
        
        calculateResidual(cublasHandler, mtxR_d, numOfA, numOfColX, newRsdl_h);
        rltvRsdl_h = newRsdl_h / orgRsdl_h; // Calculate Relative Residue
        printf("\n🫥Relative Residue: %f🫥\n\n", rltvRsdl_h);
    

        //If it is converged, then stopped.
        isStop = checkStop(cublasHandler, mtxR_d, numOfA, numOfColX, THRESHOLD);
        if(isStop)
        {
            printf("\n\n🌀🌀🌀CONVERGED🌀🌀🌀\n\n");
            break;
        }

        // Z_{i+1} <- MR_{i+1}
        startTime = myCPUTimer();
        multiply_Sprc_Den_mtx(cusparseHandler, csrMtxM, mtxR_d, numOfColX, mtxZ_d);
        endTime = myCPUTimer();
        if(counter == 1){
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
        if(crrntRank == 1){
            
            // Copy data to mtxBta to overwrite as an answer
            CHECK(cudaMemcpy(mtxBta_d, mtxQTZ_d, numOfColX * sizeof(double), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(beta_h, mtxPTQ_inv_d, crrntRank * sizeof(double), cudaMemcpyDeviceToHost));
            *beta_h *= -1.0f;

            CHECK_CUBLAS(cublasDscal(cublasHandler, numOfColX, beta_h, mtxBta_d, crrntRank));
            endTime = myCPUTimer();
            if(counter == 1){
                printf("\nbeta <- -(P'Q)^{-1} * (Q'Z_{i+1}): %f s \n", endTime - startTime);
            }

            if(debug){
                printf("\n\n~~mtxBta_d~~\n\n");
                print_mtx_clm_d(mtxBta_d, crrntRank, numOfColX);
            }

        }else{
            multiply_ngt_Den_ClmM_mtx_mtx(cublasHandler, mtxPTQ_inv_d, mtxQTZ_d, mtxBta_d, crrntRank, numOfColX, crrntRank);
            endTime = myCPUTimer();
            if(counter == 1){
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
        if(debug){
            printf("\n\n~~ (mtxZ_{i+1}_d + p * beta) ~~\n\n");
            print_mtx_clm_d(mtxZ_d, numOfA, numOfColX);
        }

        //To update matrix P
        orth_SVD(&mtxP_d, mtxZ_d, numOfA, numOfColX, crrntRank);
//        orth_QR(&mtxP_d, mtxZ_d, numOfA, crrntRank, crrntRank);
        endTime = myCPUTimer();
        if(counter == 1){
            printf("\nP_{i+1} = orth(Z_{i+1} + p * beta): %f s \n", endTime - startTime);
        }
        if(debug){
            printf("\n\n~~ mtxP_d <- orth(*)~~\n\n");
            print_mtx_clm_d(mtxP_d, numOfA, crrntRank);
            printf("\n\n= = current Rank: %d = = \n\n", crrntRank);
        }

        if(crrntRank == 0)
        {
            printf("\n\n!!!Current Rank became 0!!!\n 🔸Exit iteration🔸\n");
            break;
        }

        counter++;
    } // end of while




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

void bfbcg_BenchMark(CSRMatrix &csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX, BenchMarkBFBCG& bmBFBCG){

    double startTime, endTime; // For bench mark

    int crrntRank = numOfColX;
    const double THRESHOLD = 1e-12;
    bool isStop = false;

    double* mtxR_d = NULL; // Residual
    CSRMatrix csrMtxM = generateSparseIdentityMatrixCSR(numOfA); // Precondtion
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

    
    // //For calculating relative residual during the iteration
    // double orgRsdl_h = 0.0f; // Initial residual
    // double newRsdl_h = 0.0f; // New residual dring the iteration
    // double rltvRsdl_h = 0.0f; // Relateive resitual


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

    //Set up before iterating
    //R <- B - AX
    den_mtx_subtract_multiply_Sprc_Den_mtx(cusparseHandler, csrMtxA, mtxSolX_d, numOfColX, mtxR_d);
    // calculateResidual(cublasHandler, mtxR_d, numOfA, numOfColX, orgRsdl_h);

    //Z <- MR
    multiply_Sprc_Den_mtx(cusparseHandler, csrMtxM, mtxR_d, numOfColX, mtxZ_d);

    //P <- orth(Z), mtxZ will be freed in the function
    orth_SVD(&mtxP_d, mtxZ_d, numOfA, crrntRank, crrntRank);
    // orth_QR(&mtxP_d, mtxZ_d, numOfA, crrntRank, crrntRank);

    //Start iteration
    int counter = 1;
    const int MAX_COUNT = 100;
    while(counter < MAX_COUNT){
        
        //Q <- AP
        startTime = myCPUTimer();
        multiply_Sprc_Den_mtx(cusparseHandler, csrMtxA, mtxP_d, crrntRank, mtxQ_d);
        endTime = myCPUTimer();
        
        if(counter == 1){
            bmBFBCG.q_ap_times.push_back(endTime - startTime);
        }
        
        //(P'Q)^{-1}, save for the beta calculation
        startTime = myCPUTimer();
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxQ_d, mtxPTQ_d, numOfA, crrntRank, crrntRank);
        //Check matrix is invertible or not.
        const double CONDITION_NUM_THRESHOLD = 1000;
        double conditionNum = computeConditionNumber(mtxPTQ_d, crrntRank, crrntRank);
        assert (conditionNum < CONDITION_NUM_THRESHOLD && "\n\n!!ill-conditioned matrix A in inverse function!!\n\n");
        
        //LU factorization inverse
        // inverse_Den_Mtx(cusolverHandler, mtxPTQ_d, mtxPTQ_inv_d, crrntRank);
        
        //QR decompostion inverse
        inverse_QR_Den_Mtx(cusolverHandler, cublasHandler, mtxPTQ_d, mtxPTQ_inv_d, crrntRank);

        //(P'R)
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxR_d, mtxPTR_d, numOfA, crrntRank, numOfColX);

        //Alpha <- (P'Q)^{-1} * (P'R)
        if(crrntRank == 1){
            // Copy data to mtxAlpha to overwrite as an answer
            // CHECK(cudaMalloc((void**)&mtxAlph_d, numOfColX * sizeof(double)));
            CHECK(cudaMemcpy(mtxAlph_d, mtxPTR_d, numOfColX * sizeof(double), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(alpha_h, mtxPTQ_inv_d, sizeof(double), cudaMemcpyDeviceToHost));            
            CHECK_CUBLAS(cublasDscal(cublasHandler, numOfColX, alpha_h, mtxAlph_d, crrntRank));
            endTime = myCPUTimer();
            if(counter == 1){
                bmBFBCG.alpha_times.push_back(endTime - startTime);
            }


        }else{
            multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxPTQ_inv_d, mtxPTR_d, mtxAlph_d, crrntRank, numOfColX, crrntRank);
            endTime = myCPUTimer();
            if(counter == 1){
                bmBFBCG.alpha_times.push_back(endTime - startTime);
            }
        }

        //X_{i+1} <- x_{i} + P * alpha
        startTime = myCPUTimer();
        multiply_sum_Den_ClmM_mtx_mtx(cublasHandler, mtxP_d, mtxAlph_d, mtxSolX_d, numOfA, numOfColX, crrntRank);
        endTime = myCPUTimer();
        if(counter == 1){
            bmBFBCG.x_update_times.push_back(endTime - startTime);
        }

        //R_{i+1} <- R_{i} - Q * alpha
        startTime = myCPUTimer();
        subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandler, mtxQ_d, mtxAlph_d, mtxR_d, numOfA, crrntRank, numOfColX);
        endTime = myCPUTimer();
        if(counter == 1){
            bmBFBCG.r_update_times.push_back(endTime - startTime);
        }
        
        // calculateResidual(cublasHandler, mtxR_d, numOfA, numOfColX, newRsdl_h);
        // rltvRsdl_h = newRsdl_h / orgRsdl_h; // Calculate Relative Residue
        // printf("\n🫥Relative Residue: %f🫥\n\n", rltvRsdl_h);
    

        //If it is converged, then stopped.
        isStop = checkStop(cublasHandler, mtxR_d, numOfA, numOfColX, THRESHOLD);
        if(isStop)
        {
//            printf("\n🌀🌀🌀CONVERGED🌀🌀🌀");
            break;
        }

        // Z_{i+1} <- MR_{i+1}
        startTime = myCPUTimer();
        multiply_Sprc_Den_mtx(cusparseHandler, csrMtxM, mtxR_d, numOfColX, mtxZ_d);
        endTime = myCPUTimer();
        if(counter == 1){
            bmBFBCG.z_update_times.push_back(endTime - startTime);
        }

        //(Q'Z_{i+1})
        startTime = myCPUTimer();
        multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_d, mtxZ_d, mtxQTZ_d, numOfA, crrntRank, numOfColX);

        //beta <- -(P'Q)^{-1} * (Q'Z_{i+1})
        if(crrntRank == 1){
            
            // Copy data to mtxBta to overwrite as an answer
            CHECK(cudaMemcpy(mtxBta_d, mtxQTZ_d, numOfColX * sizeof(double), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(beta_h, mtxPTQ_inv_d, crrntRank * sizeof(double), cudaMemcpyDeviceToHost));
            *beta_h *= -1.0f;

            CHECK_CUBLAS(cublasDscal(cublasHandler, numOfColX, beta_h, mtxBta_d, crrntRank));
            endTime = myCPUTimer();
            if(counter == 1){
                bmBFBCG.beta_times.push_back(endTime - startTime);
            }

        }else{
            multiply_ngt_Den_ClmM_mtx_mtx(cublasHandler, mtxPTQ_inv_d, mtxQTZ_d, mtxBta_d, crrntRank, numOfColX, crrntRank);
            endTime = myCPUTimer();
            if(counter == 1){
                bmBFBCG.beta_times.push_back(endTime - startTime);
            }
        }



        //P_{i+1} = orth(Z_{i+1} + p * beta)
        //Z_{i+1} <- Z_{i+1} + p * beta overwrite with Sgemm function
        //Then P_{i+1} <- orth(Z_{ i+1})
        startTime = myCPUTimer();
        multiply_sum_Den_ClmM_mtx_mtx(cublasHandler, mtxP_d, mtxBta_d, mtxZ_d, numOfA, numOfColX, crrntRank);

        //To update matrix P
        orth_SVD(&mtxP_d, mtxZ_d, numOfA, numOfColX, crrntRank);
        // orth_QR(&mtxP_d, mtxZ_d, numOfA, crrntRank, crrntRank);
        endTime = myCPUTimer();
        if(counter == 1){
            bmBFBCG.p_update_times.push_back(endTime - startTime);
        }

        if(crrntRank == 0)
        {
//            printf("\n!!!Current Rank became 0!!!\n 🔸Exit iteration🔸\n");
            break;
        }

        counter++;
    } // end of while

    if(counter == MAX_COUNT){
      printf("\nNOT CONVERGED");
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
}// end of bfbcg_BenchMark

#endif // BFBCG_H