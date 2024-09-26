// Allocate and copy CSR matrix data to device
int *row_offsets_d = NULL;
int *col_indices_d = NULL;
double *vals_d = NULL;

int numRowsA = csrMtxA->_numOfRows;
int numClmsA = csrMtxA->_numOfClms;
int nnz = csrMtxA->_numOfnz;

CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

CHECK(cudaMemcpy(row_offsets_d, csrMtxA->_row_offsets, (numRowsA + 1) * sizeof(int), cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(col_indices_d, csrMtxA->_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(vals_d, csrMtxA->_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

// Create cuSPARSE descriptors
cusparseSpMatDescr_t mtxA;
cusparseDnMatDescr_t mtxB, mtxC;

// Create sparse matrix descriptor for A
CHECK_CUSPARSE(cusparseCreateCsr(
    &mtxA, numRowsA, numClmsA, nnz,
    row_offsets_d, col_indices_d, vals_d,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

// Initialize mtxB and mtxC descriptors
CHECK_CUSPARSE(cusparseCreateDnMat(
    &mtxB, numClmsA, crrntRank, numClmsA,
    NULL, CUDA_R_64F, CUSPARSE_ORDER_COL));

CHECK_CUSPARSE(cusparseCreateDnMat(
    &mtxC, numRowsA, crrntRank, numRowsA,
    NULL, CUDA_R_64F, CUSPARSE_ORDER_COL));

// Allocate buffer for cusparseSpMM
double alpha = 1.0;
double beta = 0.0;

size_t bufferSize = 0;
void *dBuffer = NULL;

// Compute buffer size using maximum possible rank
int maxRank = numOfColX; // Set this to an upper bound for crrntRank
CHECK_CUSPARSE(cusparseSpMM_bufferSize(
    cusparseHandler,
    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, mtxA, mtxB, &beta, mtxC,
    CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

CHECK(cudaMalloc(&dBuffer, bufferSize));
