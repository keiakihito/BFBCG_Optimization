// Compute (P'Q)
multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxQ_d, mtxPTQ_d, numOfA, crrntRank, crrntRank);

// Compute (P'R)
multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxP_d, mtxR_d, mtxPTR_d, numOfA, crrntRank, numOfColX);

// Solve (P'Q) * Alpha = (P'R)
startTime = myCPUTimer();
if (crrntRank == 1) {
    // Rank-one case: use scalar operations
    double a_h = 0.0;
    CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, mtxP_d, 1, mtxQ_d, 1, &a_h));

    double b_h = 0.0;
    CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, mtxP_d, 1, mtxR_d, 1, &b_h));

    double alpha_h = b_h / a_h;

    // Update X_{i+1} <- X_{i} + P * alpha
    CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &alpha_h, mtxP_d, 1, mtxSolX_d, 1));

    // Update R_{i+1} <- R_{i} - Q * alpha
    double neg_alpha_h = -alpha_h;
    CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &neg_alpha_h, mtxQ_d, 1, mtxR_d, 1));

} else {
    // General case: use linear system solver
    solve_linear_system(
        cusolverHandler, mtxPTQ_d, mtxPTR_d, mtxAlph_d, crrntRank, numOfColX);

    // Update X_{i+1} <- X_{i} + P * Alpha
    multiply_sum_Den_ClmM_mtx_mtx(cublasHandler, mtxP_d, mtxAlph_d, mtxSolX_d, numOfA, numOfColX, crrntRank);

    // Update R_{i+1} <- R_{i} - Q * Alpha
    subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandler, mtxQ_d, mtxAlph_d, mtxR_d, numOfA, crrntRank, numOfColX);
}
endTime = myCPUTimer();
if (benchmark && (1 <= counter && counter <= 6)) {
    printf("\nAlpha computation time: %f s \n", endTime - startTime);
}
