#ifndef CSRMatrix_HPP
#define CSRMatrix_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
// #include <matio.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <time.h>



#include "../functions/helper.h"
#include "../utils/checks.h"



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
              int* row_offsets, int* col_indices, double* vals)
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
CSRMatrix* generateSparseIdentityMatrixCSR(int N) {
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
    CSRMatrix* csrMtxM = new CSRMatrix(N, N, N, row_offsets, col_indices, vals);
    return csrMtxM;
}



#endif // CSRMatix_HPP