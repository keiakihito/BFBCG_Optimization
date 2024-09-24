#ifndef CSRMatrix_HPP
#define CSRMatrix_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <time.h>

#include <glm/glm.hpp>

#include "../functions/helper.h"
#include "../utils/checks.h"
#include "LargeVector.hpp"


// CSR Matrix
struct CSRMatrix{
    int _numOfRows;
    int _numOfClms;
    int _numOfnz;
    int *_row_offsets;
    int *_col_indices;
    double *_vals;

    //Constructor
    CSRMatrix(int numOfRows, int numOfClms, int numOfnz)
        : _numOfRows(numOfRows), _numOfClms(numOfClms), _numOfnz(numOfnz){
        _row_offsets = new int[numOfRows + 1];
        _col_indices = new int[numOfnz];
        _vals = new double[numOfnz];

        if(!_row_offsets || ! _col_indices || !_vals){
          std::cerr << "!!!ERROR!! Allocatioon for CSRMatrix constructor failed" << std::endl;
          exit(EXIT_FAILURE);
        }
    } // end of constructor

    //Overload constructor for Identity Sparse matrix
    CSRMatrix(int numOfRows, int numOfClms, int numOfnz,
              int *row_offsets, int *col_indices, double *vals)
        : _numOfRows(numOfRows), _numOfClms(numOfClms), _numOfnz(numOfnz),
          _row_offsets(row_offsets), _col_indices(col_indices), _vals(vals) {
    }


    // Copy Constructor
    CSRMatrix(const CSRMatrix& other)
        : _numOfRows(other._numOfRows), _numOfClms(other._numOfClms), _numOfnz(other._numOfnz) {
        _row_offsets = new int[_numOfRows + 1];
        _col_indices = new int[_numOfnz];
        _vals = new double[_numOfnz];

        //OutputIterator copy (InputIterator first, InputIterator last, OutputIterator result)
        std::copy(other._row_offsets, other._row_offsets + _numOfRows + 1, _row_offsets);
        std::copy(other._col_indices, other._col_indices + _numOfnz, _col_indices);
        std::copy(other._vals, other._vals + _numOfnz, _vals);
    }

    // Copy Assignment Operator
    CSRMatrix& operator=(const CSRMatrix& other) {
        if (this == &other) {
            return *this; // handle self-assignment
        }

        delete[] _row_offsets;
        delete[] _col_indices;
        delete[] _vals;

        _numOfRows = other._numOfRows;
        _numOfClms = other._numOfClms;
        _numOfnz = other._numOfnz;

        _row_offsets = new int[_numOfRows + 1];
        _col_indices = new int[_numOfnz];
        _vals = new double[_numOfnz];

		//OutputIterator copy (InputIterator first, InputIterator last, OutputIterator result)
        std::copy(other._row_offsets, other._row_offsets + _numOfRows + 1, _row_offsets);
        std::copy(other._col_indices, other._col_indices + _numOfnz, _col_indices);
        std::copy(other._vals, other._vals + _numOfnz, _vals);

        return *this;
    }


    //Descructor
    ~CSRMatrix(){
      delete[] _row_offsets;
      delete[] _col_indices;
      delete[] _vals;
    }
};





// Print out CSRMatrix object
void print_CSRMtx(const struct CSRMatrix &csrMtx) {
    std::cout << "\n\nnumOfRows: " << csrMtx._numOfRows
              << ", numOfClms: " << csrMtx._numOfClms
              << ", number of non zero: " << csrMtx._numOfnz << std::endl;

    std::cout << "\nrow_offsets: [ ";
    for (int wkr = 0; wkr <= csrMtx._numOfRows; ++wkr) {
        std::cout << csrMtx._row_offsets[wkr] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\ncol_indices: [ ";
    for (int wkr = 0; wkr < csrMtx._numOfnz; ++wkr) {
        std::cout << csrMtx._col_indices[wkr] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\nnon-zero values: [ ";
    for (int wkr = 0; wkr < csrMtx._numOfnz; ++wkr) {
        std::cout << std::fixed << std::setprecision(6) << csrMtx._vals[wkr] << " ";
    }
    std::cout << "]" << std::endl;
} // end of print_CSRMtx

// Print out CSRMatrix object
void print_CSRMtx(const struct CSRMatrix* csrMtx) {
    std::cout << "\n\nnumOfRows: " << csrMtx->_numOfRows
              << ", numOfClms: " << csrMtx->_numOfClms
              << ", number of non zero: " << csrMtx->_numOfnz << std::endl;

    std::cout << "\nrow_offsets: [ ";
    for (int wkr = 0; wkr <= csrMtx->_numOfRows; ++wkr) {
        std::cout << csrMtx->_row_offsets[wkr] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\ncol_indices: [ ";
    for (int wkr = 0; wkr < csrMtx->_numOfnz; ++wkr) {
        std::cout << csrMtx->_col_indices[wkr] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\nnon-zero values: [ ";
    for (int wkr = 0; wkr < csrMtx->_numOfnz; ++wkr) {
        std::cout << std::fixed << std::setprecision(6) << csrMtx->_vals[wkr] << " ";
    }
    std::cout << "]" << std::endl;
} // end of print_CSRMtx



//Convert CSR format to Dense matrix format in 1D
std::vector<double> csrToDense(const CSRMatrix &csrMtx) {
    std::vector<double> dnsMtx(csrMtx._numOfRows * csrMtx._numOfClms, 0.0);

    for(int otWkr = 0; otWkr < csrMtx._numOfRows; otWkr++) {
        for(int inWkr = csrMtx._row_offsets[otWkr]; inWkr < csrMtx._row_offsets[otWkr+1]; inWkr++) {
            dnsMtx[otWkr * csrMtx._numOfClms + csrMtx._col_indices[inWkr]] = csrMtx._vals[inWkr];
        }
    }

    return dnsMtx;
}

//Print 1D vector with 2D representation
void printDenseMatix(const std::vector<double>& dnsMtx, int numOfRows, int numOfClms){
  for(int rWkr = 0; rWkr < numOfRows; rWkr++) {
    for(int cWkr = 0; cWkr < numOfClms; cWkr++) {
      std::cout << std::fixed << std::setprecision(2) << dnsMtx[rWkr * numOfClms + cWkr] << " ";
    } // end of col loop
    std::cout << std::endl;
  } // end of row loop
}

// Generate a sparse SPD matrix in CSR format
CSRMatrix generateSparseIdentityMatrixCSR(int N) {
    int *row_offsets = new int[N+1];
    int *col_indices = new int[N];
    double *vals = new double[N];

    if (!row_offsets || !col_indices || !vals) {
        std::cerr << "Failed to allocate memory for CSR matrix." << std::endl;        exit(EXIT_FAILURE);
    }


    // Fill row_offsets, col_indices, and vals
    for (int wkr = 0; wkr < N; ++wkr) {
        row_offsets[wkr] = wkr;
        col_indices[wkr] = wkr;
        vals[wkr] = 1.0f;
    }

    // Last element of row_offsets should be the number of non-zero elements
    row_offsets[N] = N;

    //Using the overloaded constructor
    CSRMatrix csrMtx(N, N, N, row_offsets, col_indices, vals);

    return csrMtx;
}





//Convert from LargeVector<glm::mat3> to CSRMatrix
//Sub functions
// Function to check sparseity of LargeVector<glm::mat3> mtxA
double checkSparsity(const LargeVector<glm::mat3>& mtxA){
    size_t totalBlocks = mtxA.size();
    size_t nonZeroBlocks = 0;
    for(size_t i = 0; i <mtxA.size(); i++){
        if(mtxA[i] != glm::mat3(0.0f)){
            nonZeroBlocks++;
        }
    } // end of for
    return 1.0f - static_cast<double>(nonZeroBlocks) / totalBlocks;
} // end of checkSparsity


//Function to assert minimum sparsity before converting to CSR
void assertMinSparsity(const LargeVector<glm::mat3>& mtxA, const double minSparsity){
    double sparsity = checkSparsity(mtxA);
    double eps = 1e-6;

    if(sparsity + eps < minSparsity){
        std::cout << std::fixed<< std::setprecision(4) << "Sparsity + eps: " << (sparsity + eps) <<std::endl;
        std::cout << std::fixed<< std::setprecision(4) << "minSparsity: " << minSparsity <<std::endl;
    }

    assert((sparsity + eps) >= minSparsity && "Matrix A does not meet the minimum sparsity ratio");
} // end of assertMinSparsity


//Counting number of 0s
int countNumOfNonZero(const LargeVector<glm::mat3>& mtxA){
    int nnz = 0;
    bool debug = false;

    for(size_t i = 0; i <mtxA.size(); i++){
      glm::mat3 block = mtxA[i];
      for(int j = 0; j < 3; j++){
        for(int k = 0; k < 3; k++){
          if(block[j][k] != 0.0f){
            if(debug){
              std::cout << "\nmat [" << i << "]" << std::endl;
              std::cout << "block[" << j << "][" << k << "]: " << block[j][k] << std::endl;
            }
            nnz++;
          }
        }// end of k
      }// end of j
    } // end of i

    return nnz;
} // end of countNumOfNonZero


void fillCSRMatrix(const LargeVector<glm::mat3>&mtxA, CSRMatrix &csrMtxA){
  int nnzIdx = 0;
  csrMtxA._row_offsets[0] = 0;

  for(size_t i = 0; i < mtxA.size(); i++){
    const glm::mat3& block = mtxA[i];

    //Iterate over the row
    for(int j = 0; j < 3; j++){
      //Flag to check the row has non-zero value to update colmun indices
      bool isRowHasNonZeroVal = false;
      //Iterate over the column
      for(int k = 0; k < 3; k++){
        if(block[j][k] != 0.0f){
          csrMtxA._vals[nnzIdx] = block[j][k];
          csrMtxA._col_indices[nnzIdx] = i * 3 + k; // Column index in CSR format
          nnzIdx++;
          //Set the flag to update nnzVal and col_indices
          isRowHasNonZeroVal = true;
        }// end of if
      } // end of k
      if(!isRowHasNonZeroVal){
        //This case values in the row are entilry 0, which means row_indices is the same value from previous one to skip
        csrMtxA._row_offsets[i*3 + j + 1] = csrMtxA._row_offsets[i*3 + j]; //Same row value of the previous index
      }else{
      	csrMtxA._row_offsets[i*3 + j + 1] = nnzIdx;
      }

    }// enf of j

  }// end of i
} // end of fillCSRMatrix


void fillCSRMatrix(const LargeVector<glm::mat3>&mtxA, CSRMatrix* csrMtxA){
  int nnzIdx = 0;
  csrMtxA->_row_offsets[0] = 0;

  for(size_t i = 0; i < mtxA.size(); i++){
    const glm::mat3& block = mtxA[i];

    //Iterate over the row
    for(int j = 0; j < 3; j++){
      //Flag to check the row has non-zero value to update colmun indices
      bool isRowHasNonZeroVal = false;
      //Iterate over the column
      for(int k = 0; k < 3; k++){
        if(block[j][k] != 0.0f){
          csrMtxA->_vals[nnzIdx] = block[j][k];
          csrMtxA->_col_indices[nnzIdx] = i * 3 + k; // Column index in CSR format
          nnzIdx++;
          //Set the flag to update nnzVal and col_indices
          isRowHasNonZeroVal = true;
        }// end of if
      } // end of k
      if(!isRowHasNonZeroVal){
        //This case values in the row are entilry 0, which means row_indices is the same value from previous one to skip
        csrMtxA->_row_offsets[i*3 + j + 1] = csrMtxA->_row_offsets[i*3 + j]; //Same row value of the previous index
      }else{
      	csrMtxA->_row_offsets[i*3 + j + 1] = nnzIdx;
      }

    }// enf of j

  }// end of i
} // end of fillCSRMatrix




//Convert from LargeVector<glm::mat3>& mtxA
CSRMatrix* convertLargeVectorToCSRMtx(const LargeVector<glm::mat3>& mtxA){

  	int numOfA = mtxA.size();

    //Step1: Count non-zero value
    int nnz = countNumOfNonZero(mtxA);

    //Step2: Create
    //Make sure the actual Row and Colmun size in the <glm::mat3>mtx A is (mtxA.size() * 3)
    CSRMatrix* csrMtxA = new CSRMatrix(numOfA*3, numOfA*3, nnz);

    //Step3: Fill index and non zero values
    fillCSRMatrix(mtxA, csrMtxA);

    return csrMtxA;
} // end of convertLargeVectorToCSRMtx

void convertCSRtoMatlab(const CSRMatrix* csrMtxA, const std::string& fileName){
  	bool debug = false;
  	std::vector<int> rowIndices;
    std::vector<int> colIndices(csrMtxA->_col_indices, csrMtxA->_col_indices + csrMtxA->_numOfnz);
    std::vector<double> vals(csrMtxA->_vals, csrMtxA->_vals+csrMtxA->_numOfnz);

    //Convert row offsets to row indices
    for(int i = 0; i < csrMtxA->_numOfRows; i++){
      for(int j = csrMtxA->_row_offsets[i]; j < csrMtxA->_row_offsets[i+1]; j++){
       	//Matlab uses 1-base indexing
        rowIndices.push_back(i+1);
      }
    }// end of nested loop

    //Write to file in Matlab format
    std::ofstream file(fileName);
    if(file.is_open()){

      file << "rowIndices = [";
      for(size_t i = 0; i < rowIndices.size(); i++){
        file << rowIndices[i];
        if(i < rowIndices.size()-1){file << " ";}
      } // end of for
      file << "]; \n";

      file << "colIndices = [";
      for(size_t i = 0; i < colIndices.size(); i++){
        file << colIndices[i] + 1; // Note Matlab uses 1-based index
        if(i < colIndices.size()-1){file << " ";}
      } // end of for
      file << "]; \n";

      file << "vals = [";
      for(size_t i = 0; i < vals.size(); i++){
        file << vals[i]; // Note Matlab uses 1-based index
        if(i < vals.size()-1){file << " ";}
      } // end of for
      file << "]; \n";

      file.close();

      if(debug){
        print_CSRMtx(csrMtxA);
      }

    }else{
      std::cerr << "Unable to open file for writing." << std::endl;
    }
} // end of convertCSRtoMatlab


#endif // CSRMatix_HPP