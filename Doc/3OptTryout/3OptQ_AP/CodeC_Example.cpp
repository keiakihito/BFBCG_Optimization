// Update dimensions in descriptors if crrntRank changes
if (crrntRank_changed) {
    CHECK_CUSPARSE(cusparseDnMatSetDimensions(mtxB, numClmsA, crrntRank, numClmsA));
    CHECK_CUSPARSE(cusparseDnMatSetDimensions(mtxC, numRowsA, crrntRank, numRowsA));
}

// Q <- A * P
startTime = myCPUTimer();
multiply_Sprc_Den_mtx(
    cusparseHandler,
    mtxA,
    mtxB,
    mtxC,
    mtxP_d,  // Pointer to P
    mtxQ_d,  // Pointer to Q
    dBuffer);
endTime = myCPUTimer();
if (benchmark && (1 <= counter && counter <= 6)) {
    printf("\nQ <- AP computation time: %f s \n", endTime - startTime);
}
