// Compute (Q'Z_{i+1})
multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_d, mtxZ_d, mtxQTZ_d, numOfA, crrntRank, numOfColX);

// Multiply mtxQTZ_d by -1 to get RHS of the linear system
double minus_one = -1.0;
CHECK_CUBLAS(cublasDscal(cublasHandler, crrntRank * numOfColX, &minus_one, mtxQTZ_d, 1));

// Solve (P'Q) * Beta = - (Q'Z_{i+1})
startTime = myCPUTimer();
if (crrntRank == 1) {
    // Rank-one case: use scalar operations
    double a_h = 0.0;
    CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, mtxP_d, 1, mtxQ_d, 1, &a_h));

    double c_h = 0.0;
    CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, mtxQ_d, 1, mtxZ_d, 1, &c_h));

    double beta_h = -c_h / a_h;

    // Update Z_{i+1} <- Z_{i+1} + P * beta
    CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &beta_h, mtxP_d, 1, mtxZ_d, 1));

} else {
    // General case: use linear system solver
    solve_linear_system(
        cusolverHandler, mtxPTQ_d, mtxQTZ_d, mtxBta_d, crrntRank, numOfColX);

    // Update Z_{i+1} <- Z_{i+1} + P * Beta
    multiply_sum_Den_ClmM_mtx_mtx(cublasHandler, mtxP_d, mtxBta_d, mtxZ_d, numOfA, numOfColX, crrntRank);
}
endTime = myCPUTimer();
if (benchmark && (1 <= counter && counter <= 6)) {
    printf("\nBeta computation time: %f s \n", endTime - startTime);
}
