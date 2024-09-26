for (int i = 0; i < currentRank; ++i) {
    CHECK_CUBLAS(cublasDscal(
        cublasHandler,
        numOfRow,
        norms_d + i,
        mtxY_d + i * numOfRow,
        1));
}
