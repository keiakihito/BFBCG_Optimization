for (int i = 0; i < currentRank; ++i) {
    CHECK_CUBLAS(cublasDnrm2(
        cublasHandler,
        numOfRow,
        mtxY_d + i * numOfRow,
        1,
        norms_d + i));
}
