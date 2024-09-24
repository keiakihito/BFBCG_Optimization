#ifndef ORTH_QR_H
#define ORTH_QR_H


#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <magma_v2.h>
#include <magma_lapack.h>

// helper function CUDA error checking and initialization
#include "helper.h"
#include "cuBLAS_util.h"
#include "cuSOLVER_util.h"

//Input: double* mtxQ_trnc_d, double* mtxZ_d, int number of Row, int number Of column, int & currentRank
//Process: the function extracts orthonormal set from the matrix Z
//Output: double* mtxQ_trnc_d, the orthonormal set of matrix Z with significant column vectors
void orth_QR(double** mtxQ_trnc_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank);

//Input: singluar values, int currnet rank, double threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(double* mtxR_d, int numOfRow, int currentRank, double threashold);

//Process: count the number of tau values
int getNumOfTau(double* tau_h, int numOfClm);

/*
Personal Note: How to Run magma in NCSA
1. Srun
2. Set path
export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64:$LD_LIBRARY_PATH

3. Run orth_QRTest.cu
nvcc orth_QRTest.cu -o orth_QRTest -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64 -lcudart -lcublas -lcusolver -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openblas-0.3.25-5yvxjnl/lib -lmagma -lopenblas

4. ./orth_QRTest
 */
void orth_QR(double** mtxQ_trnc_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank)
{
    magma_init();
    assert(numOfClm < numOfRow && "The nuber of row should be larger the number of clomn, matrix should be tall and skiny");
    bool debug = false;
    const double THREASHOLD = 1e-6;

    magmaDouble_ptr mtxZ_cpy_d = NULL;
    magmaDouble_ptr mtxQ_d = NULL;
    magmaDouble_ptr tau_d = NULL;

    magma_int_t lda = numOfRow; // leading dimenstion
    magma_int_t mn = numOfClm; // numOfClm < numOfRow
    magma_int_t lwork = 0;
    magma_int_t nb = magma_get_dgeqp3_nb(lda, mn); // Block size
    magma_int_t info = 0;
    magma_int_t* ipiv = NULL; //pivotig information
    magma_int_t numOfTau = 0;

    magmaDouble_ptr tau_h = NULL;
    magmaDouble_ptr dT = NULL;
    magmaDouble_ptr work_d = NULL;


    //(1) Allocate memory
    CHECK_MAGMA(magma_dmalloc(&mtxZ_cpy_d, lda * mn));
    CHECK_MAGMA(magma_dmalloc(&mtxQ_d, lda * lda));

    // Allocate pivot value in host and initialize 0 surely.
    ipiv = (int*)calloc(mn, sizeof(int));
    check_allocation(ipiv);

    // Allocate tau values in devide side and initialize with calloc.
    tau_h = (double*)calloc(numOfClm, sizeof(double));
    check_allocation(tau_h);
    CHECK_MAGMA(magma_dmalloc(&tau_d, mn));
    CHECK(cudaMemcpy(tau_d, tau_h, mn * sizeof(magmaDouble_ptr), cudaMemcpyHostToDevice));
    
    CHECK_MAGMA(magma_dmalloc(&dT, (2 * mn + magma_roundup(mn, 32)) * nb));


    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxZ_cpy_d, mtxZ_d, lda * mn * sizeof(magmaDouble_ptr), cudaMemcpyDeviceToDevice));

    //(3) Calculate lwork based on the documentation
    lwork = (mn + 1) * nb + 2 * mn;
    lwork = max( lwork, lda*mn + mn );
    if(debug){
        printf("\n\n~~mtxZ_cpy_d~~\n\n");
        print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
        printf("\nlda: %d", lda);
        printf("\nmn: %d", mn);
        printf("\nnb: %d", nb);
        printf("\nlwork: %d", lwork);
    }

    //(4) Allocate workspace 
    CHECK_MAGMA(magma_dmalloc(&work_d, lwork));

    if(debug){
        printf("\n\n= = Before magma_dgeqp3_gpu = = \n\n");
        printf("\n\n~~mtxZ_cpy_d~~\n\n");
        print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
        printf("\n\nipiv: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%d,", ipiv[i]);
        }
        printf("\n\ntau_d: \n");
        print_vector(tau_d, numOfClm);
    }
    

    //(5) Perform QR decompostion with column pivoting
    //mtxZ_cpy_d contains the R matrix in its upper triangular part,
    //the orthogonal matrix Q is represented implicitly by the Householder vectors stored in the lower triangular part of mtxZ_cpy_d and in tau_d, scalor.
    CHECK_MAGMA(magma_dgeqp3_gpu(lda, mn, mtxZ_cpy_d, lda, ipiv, tau_d, work_d, lwork, &info));
    
    if(debug){
        printf("\n= = After magma_dgeqp3_gpu = = \n\n");
        printf("\n\nipiv: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%d, ", ipiv[i]);
        }
        printf("\n\ntau_d: \n");
        print_vector(tau_d, numOfClm);

        printf("\n\n~~mtxZ_cpy_d~~\n-the R matrix in its upper triangular part\n-the orthogonal matrix Q is represented implicitly by the Householder vectors stored in the lower triangular\n\n");
        print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
    }

    //Copy tau values in host for magma_dorgqr_gpu later
    CHECK(cudaMemcpy(tau_h, tau_d, numOfClm * sizeof(double), cudaMemcpyDeviceToHost));

    if(debug){
        printf("\n\ntau_h: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%f\n", tau_h[i]);
        }
    }
    
    
    //(6) Iterate diagonal elements matrix R that is upper trianguar part of mtxZ_cpy_d to find rank
    currentRank = setRank(mtxZ_cpy_d, lda, currentRank, THREASHOLD);
    numOfTau = getNumOfTau(tau_h, mn); // Set up for magma_dorgqr_gpu
    nb = magma_get_dgeqrf_nb(lda, mn); // Block size for magma_dorgqr_gpu
    lwork = mn*nb; // Recalculate lwork for magma_dorgqr_gpu

    //Reallocate work_d space
    CHECK_MAGMA(magma_free(work_d));
    CHECK_MAGMA(magma_dmalloc(&work_d, lwork));

    if(debug){
        printf("\n\n= = Before dorgqr_gpu = = \n");
        printf("\nnumber of tau: %d", numOfTau);
        printf("\ncurrent rank: %d", currentRank);
        printf("\nnb: %d", nb);
        printf("\nlwork: %d", lwork);
    }
    
    //(7) Construct matrix Q with tau values and upper triangular R and householder lower triangular in mtxZ_cpy_d
    CHECK_MAGMA(magma_dorgqr_gpu(lda, mn, numOfTau, mtxZ_cpy_d, lda, tau_h, dT, nb, &info));
    
    //(8) Copy matrix Q to the mtxQ_trnc_d with significant number of column vectors calculated by upper triangular diagonal values
    CHECK(cudaMemcpy(mtxQ_d, mtxZ_cpy_d, numOfRow * currentRank * sizeof(magmaDouble_ptr), cudaMemcpyDeviceToDevice));

    if(debug){
        printf("\n\n= = After magma_dorgqr_gpu = = \n");
        printf("\n\n~~mtxQ_d truncated~~\n\n");
        print_mtx_clm_d(mtxQ_d, numOfRow, currentRank);
    }
    


    //(9)Pass the address to the provided pointer, updating orhtonomal set
    CHECK(cudaFree(*mtxQ_trnc_d));
    *mtxQ_trnc_d = NULL;
    *mtxQ_trnc_d = mtxQ_d;

    //(10) Clean up
    CHECK_MAGMA(magma_free(mtxZ_cpy_d));
    CHECK_MAGMA(magma_free_cpu(ipiv));
    CHECK_MAGMA(magma_free_cpu(tau_h));
    CHECK_MAGMA(magma_free(tau_d));
    CHECK_MAGMA(magma_free(work_d));
    CHECK_MAGMA(magma_free(dT));

    magma_finalize();

} // end of orth_QR




//Process: Check diagoal values in matrix R If it is larger than threashould, increment rank.
//Output: int newRank
int setRank(double* mtxR_d, int numOfRow, int currentRank, double threashold)
{   
    bool debug = false;
    int newRank = 0;


    double* mtxR_h = (double*)malloc(numOfRow * currentRank * sizeof(double));
    check_allocation(mtxR_h);

    CHECK(cudaMemcpy(mtxR_h, mtxR_d, numOfRow * currentRank * sizeof(double), cudaMemcpyDeviceToHost));


    if(debug){
        printf("\n\n~~In setRank copy matrix R_h ~~\n");
        for(int i = 0; i < numOfRow; i++){
            for(int j = 0; j< currentRank; j++){
                printf("%f ", mtxR_h[i + j * numOfRow]);
            }
            printf("\n");
        }
    }

    for(int wkr = 0; wkr < currentRank; wkr++){
        if(debug){
            printf("\nDiagonal Elment: %f", mtxR_h[wkr * numOfRow + wkr]);
        }
        if(fabs(mtxR_h[wkr * numOfRow + wkr]) > threashold){
            newRank++;
        } // end of if
    } // end of for

    free(mtxR_h);
    assert(newRank <= currentRank && "new rank should be less than equal to current rank");

    return newRank;
}


//Process count number of tau values
int getNumOfTau(double* tau_h, int numOfClm){
    int numOfTau = 0;
    for(int i = 0; i < numOfClm; i++){
        if(tau_h[i] != 0.0){
            numOfTau++;
        }
    }
    return numOfTau;
}



#endif // ORTH_QR_H