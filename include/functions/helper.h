#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstdlib>
#include <sys/time.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>


// helper function CUDA error checking and initialization
#include "../utils/checks.h"  
#include "../struct/CSRMatrix.hpp"



// Time tracker for each iteration
double myCPUTimer();

template<typename T>
void print_vector(const T *d_val, int size);

template<typename T>
void print_mtx_clm_d(const T *mtx_d, int numOfRow, int numOfClm);

template<typename T>
void printVecCPU(const T *vec_x_ptr, size_t size);

void initializeRandom(double mtxB_h[], int numOfRow, int numOfClm);



// = = = CPU = = ==
//Convert from LargeVector<glm::mat3> to CSRMatrix
//double checkSparsity(const LargeVector<glm::mat3>& mtxA);
//void assertMinSparsity(const LargeVector<glm::mat3>& mtxA, const double minSparsity);

template<typename T>
void printVecCPU(const T *vec_x_ptr, size_t size){
  for(size_t i = 0; i < size; i++){
    printf("%f\n", static_cast<double>(vec_x_ptr[i]));
  }
}

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


//Copy function
double* copyVector(double* sourceVec, int size){
	//Allocate memory for the destination vector
	double* destVec = new double[size];
	if(destVec == NULL){
		perror("Failed to allocate memoery for vector copy");
		return NULL;
	}

	//Copy the contents from sourceVec to destVec
	memcpy(destVec, sourceVec, size*sizeof(double));

	return destVec;
}

//Import CSRMtx object from .mm file
// Function to import a CSRMatrix from a Matrix Market file
CSRMatrix* importCSRMatrixFromMM(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    int rows, cols, nonzeros;

    // Skip the header line
    std::getline(file, line);

    // Read matrix dimensions and non-zero entries
    file >> rows >> cols >> nonzeros;

    // Create the CSRMatrix object
    CSRMatrix* csrMtxA = new CSRMatrix(rows, cols, nonzeros);

    // Temporary arrays to hold the COO format data
    std::vector<int> row_counts(rows + 1, 0);

    // Read the matrix entries
    for (int i = 0; i < nonzeros; i++) {
        int row, col;
        double value;
        file >> row >> col >> value;

        // Convert from 1-based to 0-based indexing
        row--;
        col--;

        // Fill the CSRMatrix arrays
        csrMtxA->_col_indices[i] = col;
        csrMtxA->_vals[i] = value;
        row_counts[row]++;
    }

    // Build the row_offsets array
    csrMtxA->_row_offsets[0] = 0;
    for (int i = 1; i <= rows; i++) {
        csrMtxA->_row_offsets[i] = csrMtxA->_row_offsets[i - 1] + row_counts[i - 1];
    }

    file.close();
    return csrMtxA;
}

//Import  vector_b.mm file
// Function to import a 1D column vector from a Matrix Market file and return double*
double* importVectorFromMM(const std::string& filename, int* size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    int rows, cols, nonzeros;

    // Skip the header line
    std::getline(file, line);

    // Read matrix dimensions and non-zero count
    file >> rows >> cols >> nonzeros;

    if (cols != 1) {
        std::cerr << "Error: the file does not contain a 1-column vector." << std::endl;
        return nullptr;
    }

    // Initialize a double* array to hold the vector
    *size = rows; // The size of the vector
    double* vecB_h = new double[*size];

    // Initialize the vector to zeros (in case it's sparse)
    for (int i = 0; i < *size; i++) {
        vecB_h[i] = 0.0;
    }

    // Read the matrix entries and fill the array
    for (int i = 0; i < nonzeros; i++) {
        int row, col;
        double value;
        file >> row >> col >> value;

        // Convert from 1-based to 0-based indexing
        row--;

        // Assign value to the correct index in the vector
        vecB_h[row] = value;
    }

    file.close();
    return vecB_h;
}







#endif // HELPER_H