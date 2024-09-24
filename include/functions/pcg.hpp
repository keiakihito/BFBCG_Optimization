#ifndef PCG_HPP
#define PCG_HPP


// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include<sys/time.h>
#include<glm/glm.hpp>

//Utilities
#include "../utils/checks.h"
#include "helper.h"
#include "cuBLAS_util.h"
#include "cuSPARSE_util.h"
#include "../struct/LargeVector.hpp"
#include "../struct/CSRMatrix.hpp"




//Process: Cojugate Gradient to solve LargeVector<glm::mat3> vecX
//Output: LargetVector<glm::mat3> vecX
void solveCG_GPU(LargeVector<glm::mat3> &mtxA_h, LargeVector<glm::mat3> &vecX_h, LargeVector<glm::mat3> &vecB_h);

//Input:
//Process: Conjugate Gradient with cuda GPU
//Output: vecSolX
void pcg(struct CSRMatrix *csrMtxA, double *vecSolX_h, double *vecB_h, int numOfA);



void solveCG_GPU(LargeVector<glm::mat3> &mtxA_h, LargeVector<glm::vec3> &largeVecV, LargeVector<glm::vec3> &largeVecB){
    //(0) Convert LargeVector<glm::mat3> to CSRMatrix object
    struct CSRMatrix* csrMtxA = convertLargeVectorToCSRMtx(mtxA_h);
    double* vecX_h = convertLargeVecToDoublePtr(largeVecV);
    double* vecB_h = convertLargeVecToDoublePtr(largeVecB);

    //(1) Allocate memoery
    double* vecX_d = nullptr;
    double* vecB_d = nullptr;

    CHECK(cudaMalloc((void**)&vecX_d, largeVecV.size() * sizeof(double) * 3));
    CHECK(cudaMalloc((void**)&vecB_d, largeVecB.size() * sizeof(double) * 3));

    //(2) Copy Data from CPU to GPU
    CHECK(cudaMemcpy(vecX_d, vecX_h, largeVecV.size() * sizeof(double) * 3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vecB_d, vecB_h, largeVecB.size() * sizeof(double) * 3, cudaMemcpyHostToDevice));

    //(3) Call pcg to utilize cuda functions
    pcg(csrMtxA, vecX_d, vecB_d, mtxA_h.size()*3);

    //(4) Copy solution vector x to <glm::vec3> &largeVecV with overwriting
    CHECK(cudaMemcpy(vecX_h, vecX_d, largeVecV.size() * sizeof(double) * 3, cudaMemcpyDeviceToHost));
    largeVecV = convertDoublePtrToLargeVec(vecX_h, largeVecV.size()*3);

    //(5) Free GPU memory
    CHECK(cudaFree(vecX_d));
    CHECK(cudaFree(vecB_d));
    delete[] vecX_h;
    delete[] vecB_h;
    delete csrMtxA;
} // end of pcg






void pcg(struct CSRMatrix *csrMtxA, double *vecSolX_d, double *vecB_d, int numOfA)
{
    double startTime, endTime;
    bool debug = false;
    bool benchMark = true; // Check benchmark for the first iteration each operation
    const double THRESHOLD = 1e-8;

    double *r_d = NULL; // Residual
    double *s_d = NULL; // For s <- M * r and delta <- r' * s
    struct CSRMatrix *csrMtxM = generateSparseIdentityMatrixCSR(numOfA); // Precondtion
    double *dirc_d = NULL; // Direction
    double *q_d = NULL; // Vector Ad
    double dot = 0.0f; // temporary val for d^{T} *q to get aplha

    //Using for cublas functin argument
    double alpha = 1.0;


    double initial_delta = 0.0;
    double delta_new = 0.0;
    double delta_old = 0.0;
    double relative_residual = 0.0;

    // In CG iteration alpha and beta
    double alph = 0.0f;
    double ngtAlph = 0.0f;
    double bta = 0.0f;



    //Crete handler
    cublasHandle_t cublasHandler = NULL;
    cusparseHandle_t cusparseHandler = NULL;


    CHECK_CUBLAS(cublasCreate(&cublasHandler));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandler));

    //(1) Allocate space in global memory
    CHECK(cudaMalloc((void**)&r_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&s_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&dirc_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&q_d, sizeof(double) * numOfA));

    //(2) Copy from host to device
    CHECK(cudaMemcpy(r_d, vecB_d, sizeof(double) * numOfA, cudaMemcpyDeviceToDevice));



    //(5) Iteration
    /* 💫💫💫Begin CG💫💫💫 */
    //Setting up the initial state.


    //r = b - Ax
    den_vec_subtract_multiplly_Sprc_Den_vec(cusparseHandler, csrMtxA, vecSolX_d, r_d);
    if(debug){
        printf("\n\nr_{0} = \n");
        print_vector(r_d, numOfA);
    }


    //Set d <- M * r;
    //M is Identity matrix for the place holder of precondition
    // CHECK(cudaMemcpy(dirc_d, r_d, N * sizeof(double), cudaMemcpyDeviceToDevice));
    multiply_Sprc_Den_vec(cusparseHandler, csrMtxM, r_d, dirc_d);
    if(debug){
        printf("\n\nd <- M * r");
        printf("\n\n~~vector d~~\n");
        print_vector(dirc_d, numOfA);
    }


    //delta_{new} <- r^{T} * d
    // Compute the squared norm of the initial residual vector r (stored in r1).
    CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, r_d, 1, dirc_d, 1, &delta_new));
    //Save it for the relative residual calculation.
    initial_delta = delta_new;
    if(debug){
        printf("\n\ndelta_new{0} = \n %f\n ", initial_delta);
    }


    int counter = 1; // counter
    const int MAX_ITR = 10000;

    while(counter <= MAX_ITR){

      	if((benchMark && 0 <= counter && counter <= 6) || debug){
        	printf("\n\n💫💫💫= = = Iteraion %d= = = 💫💫💫\n", counter);
      	}



        //q <- Ad
        startTime = myCPUTimer();
        multiply_Sprc_Den_vec(cusparseHandler, csrMtxA, dirc_d, q_d);
        endTime = myCPUTimer();
        if(benchMark && 0 <= counter && counter <= 6){
            printf("\nq <- Ad: %f s \n", endTime - startTime);
        }

        if(debug){
            printf("\nq = \n");
            print_vector(q_d, numOfA);
        }


        //dot <- d^{T} * q
        startTime = myCPUTimer();
        CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, dirc_d, 1, q_d, 1, &dot));
        //✅
        if(debug){
            printf("\n\n~~(d'* q)~~\n %f\n", dot);
        }


        //alpha(a) <- delta_{new} / dot // dot <- d^{T} * q
        alph = delta_new / dot;
        endTime = myCPUTimer();
        if(benchMark && 0 <= counter && counter <= 6){
            printf("\nalpha <- delta_{new} / d^{T} * q: %f s \n", endTime - startTime);
        }

        if(debug){
            printf("\nalpha = %f\n", alph);
        }


        //x_{i+1} <- x_{i} + alpha * d_{i}
        startTime = myCPUTimer();
        CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &alph, dirc_d, 1, vecSolX_d, 1));
        endTime = myCPUTimer();
        if(benchMark && 0 <= counter && counter <= 6){
            printf("\nx_{i+1} <- x_{i} + alpha * d_{i}: %f s \n", endTime - startTime);
        }

        if(debug){
            printf("\nx_sol = \n");
            print_vector(vecSolX_d, numOfA);
        }


        if(counter % 50 == 0){
            //r <- b -Ax Recompute
            CHECK(cudaMemcpy(r_d, vecB_d, sizeof(double) * numOfA, cudaMemcpyHostToDevice));
            den_vec_subtract_multiplly_Sprc_Den_vec(cusparseHandler, csrMtxA, vecSolX_d, r_d);
            if(debug){
                printf("\n\nr_{0} = \n");
                print_vector(r_d, numOfA);
            }
        }else{
            //r_{i+1} <- r_{i} -alpha*q
            startTime = myCPUTimer();
            ngtAlph = -alph;
            CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &ngtAlph, q_d, 1, r_d, 1));
            endTime = myCPUTimer();
            if(benchMark && 0 <= counter && counter <= 6){
                printf("\nr_{i+1} <- r_{i} - alpha * q: %f s \n", endTime - startTime);
            }

            if(debug){
                printf("\n\nr = \n");
                print_vector(r_d, numOfA);
            }

        }

        //s <- M * r
        startTime = myCPUTimer();
        multiply_Sprc_Den_vec(cusparseHandler, csrMtxM, r_d, s_d);
        endTime = myCPUTimer();
        if(benchMark && 0 <= counter && counter <= 6){
                printf("\ns <- M * r: %f s \n", endTime - startTime);
            }

        // delta_old <- delta_new
        delta_old = delta_new;

        // delta_new <- r' * s
        // bta <- delta_new / delta_old
        startTime = myCPUTimer();
        CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, r_d, 1, s_d, 1, &delta_new));
        bta = delta_new / delta_old;
        endTime = myCPUTimer();
        if(benchMark && 0 <= counter && counter <= 6){
            printf("\nbeta <- r' * s / delta_old: %f s \n", endTime - startTime);
        }
        if(debug){
            printf("\n\ndelta_new = %f\n", delta_new);
            printf("\nbta = %f\n", bta);
        }


        relative_residual = sqrt(delta_new)/sqrt(initial_delta);
        if(debug){
        	printf("\n\n🫥Relative residual = %f🫥\n", relative_residual);
        }


        if(sqrt(delta_new) < THRESHOLD){
        //    printf("\n\n🌀🌀🌀CONVERGED🌀🌀🌀\n\n");
            printf("\n\n🫥Relative residual = %f🫥\n", relative_residual);
            break;
        }


        //d <- s + ßd
        startTime = myCPUTimer();
        CHECK_CUBLAS(cublasDscal(cublasHandler, numOfA, &bta, dirc_d, 1)); //d <- ßd
        CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &alpha, s_d, 1, dirc_d, 1)); // d <- s + d
        endTime = myCPUTimer();
        if(benchMark && (0 <= counter && counter <= 6)){
            printf("\nd_{i+1} <- s + d_{i} * beta: %f s \n", endTime - startTime);
        }
        if(debug){
            printf("\nd = \n");
            print_vector(dirc_d, numOfA);
        }

        counter++;
    } // end of while

	if(counter == MAX_ITR){
      printf("\n\n!!!CG_GPU not converted and reached %d iterations!!!\n\n", MAX_ITR);
	}


    //(6) Free the GPU memory after use
    CHECK_CUBLAS(cublasDestroy(cublasHandler));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandler));
    CHECK(cudaFree(r_d));
    CHECK(cudaFree(s_d));
    CHECK(cudaFree(dirc_d));
    CHECK(cudaFree(q_d);)
    free(csrMtxM);

}// end of pcg






#endif // PCG_HPP