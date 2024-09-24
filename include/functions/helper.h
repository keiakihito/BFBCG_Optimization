#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstdlib>
#include <sys/time.h>
#include <cassert>

// helper function CUDA error checking and initialization
#include "../utils/checks.h"  
#include "../struct/CSRMatrix.hpp"
#include "../struct/LargeVector.hpp"


// Time tracker for each iteration
double myCPUTimer();

template<typename T>
void print_vector(const T *d_val, int size);

template<typename T>
void print_mtx_clm_d(const T *mtx_d, int numOfRow, int numOfClm);

void initializeRandom(double mtxB_h[], int numOfRow, int numOfClm);


// = = = CPU = = ==
//Convert from LargeVector<glm::mat3> to CSRMatrix
double checkSparsity(const LargeVector<glm::mat3>& mtxA);
void assertMinSparsity(const LargeVector<glm::mat3>& mtxA, const double minSparsity);





//= = = = =GPU = = = = ==
// Time tracker for each iteration
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

template<typename T>
void check_allocation(const T *val_h){
    if(val_h == NULL){
        fprintf(stderr, "\n!!Failed to allocate host memory!!\n");
        exit(EXIT_FAILURE);
    }
}

template<typename T>
void print_vector(const T *d_val, int size) {
    // Allocate memory on the host
    T *check_r = (T *)malloc(sizeof(T) * size);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    // cudaError_t err = cudaMemcpy(check_r, d_val, size * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK(cudaMemcpy(check_r, d_val, size * sizeof(T), cudaMemcpyDeviceToHost));
    // if (err != cudaSuccess) {
    //     printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    //     free(check_r);
    //     return;
    // }
    // Print the values to check them
    for (int i = 0; i < size; i++) {
            printf("%.10f \n", check_r[i]);
    }
    

    // Free allocated memory
    free(check_r);
} // print_vector



//Print matrix column major
template <typename T>
void print_mtx_clm_d(const T *mtx_d, int numOfRow, int numOfClm){
    //Allocate memory oh the host
    T *check_r = (T *)malloc(sizeof(T) * numOfRow * numOfClm);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    CHECK(cudaMemcpy(check_r, mtx_d, numOfRow * numOfClm * sizeof(T), cudaMemcpyDeviceToHost));
    // cudaError_t err = cudaMemcpy(check_r, mtx_d, numOfRow * numOfClm * sizeof(T), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //     printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    //     free(check_r);
    //     return;
    // }

    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++){
        for(int clWkr = 0; clWkr < numOfClm; clWkr++){
            printf("%f ", check_r[clWkr*numOfRow + rwWkr]);
        }// end of column walker
        printf("\n");
    }// end of row walker
} // end of print_mtx_h




//Initialize random values between -1 and 1
void initializeRandom(double mtxB_h[], int numOfRow, int numOfClm)
{
    srand(time(NULL));

    for (int wkr = 0; wkr < numOfRow * numOfClm; wkr++){
        //Generate a random double between -1 and 1
        double rndVal = ((double)rand() / RAND_MAX) * 2.0f - 1.0f;
        mtxB_h[wkr] = rndVal;
    }
} // end of initializeRandom













#endif // HELPER_H