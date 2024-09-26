// Destroy cuSPARSE descriptors
CHECK_CUSPARSE(cusparseDestroySpMat(mtxA));
CHECK_CUSPARSE(cusparseDestroyDnMat(mtxB));
CHECK_CUSPARSE(cusparseDestroyDnMat(mtxC));

// Free device memory
CHECK(cudaFree(dBuffer));
CHECK(cudaFree(row_offsets_d));
CHECK(cudaFree(col_indices_d));
CHECK(cudaFree(vals_d));
