void multiply_Sprc_Den_mtx(
    cusparseHandle_t cusparseHandler,
    cusparseSpMatDescr_t mtxA,
    cusparseDnMatDescr_t mtxB,
    cusparseDnMatDescr_t mtxC,
    double *dnsMtxB_d, // Device pointer to P
    double *dnsMtxC_d, // Device pointer to Q
    void *dBuffer
) {
    double alpha = 1.0;
    double beta = 0.0;

    // Update pointers in descriptors
    CHECK_CUSPARSE(cusparseDnMatSetValues(mtxB, dnsMtxB_d));
    CHECK_CUSPARSE(cusparseDnMatSetValues(mtxC, dnsMtxC_d));

    // Perform sparse-dense matrix multiplication
    CHECK_CUSPARSE(cusparseSpMM(
        cusparseHandler,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mtxA, mtxB, &beta, mtxC,
        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
}
