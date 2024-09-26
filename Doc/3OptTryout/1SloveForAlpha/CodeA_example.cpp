// Function to solve linear system A * X = B
void solve_linear_system(
    cusolverDnHandle_t cusolverHandler,
    double* mtxA_d,    // Device pointer to matrix A (P'Q)
    double* mtxB_d,    // Device pointer to matrix B (P'R)
    double* mtxX_d,    // Device pointer to solution X (Alpha)
    int N,             // Dimension of A (N x N)
    int NRHS           // Number of right-hand sides (columns of B)
) {
    int* devIpiv = NULL;  // Pivot indices
    int* devInfo = NULL;  // Info about the success of the solve
    int lwork = 0;
    double* work_d = NULL;

    // Allocate memory for pivot indices and info
    CHECK(cudaMalloc((void**)&devIpiv, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    // Query workspace size for LU factorization
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cusolverHandler, N, N, mtxA_d, N, &lwork));
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(double)));

    // Perform LU factorization
    CHECK_CUSOLVER(cusolverDnDgetrf(
        cusolverHandler, N, N, mtxA_d, N, work_d, devIpiv, devInfo));

    // Check if the factorization was successful
    int devInfo_h = 0;
    CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) {
        printf("LU factorization failed with info = %d\n", devInfo_h);
    }

    // Solve the system A * X = B
    CHECK_CUSOLVER(cusolverDnDgetrs(
        cusolverHandler, CUBLAS_OP_N, N, NRHS, mtxA_d, N, devIpiv, mtxB_d, N, devInfo));

    // Copy the result from mtxB_d to mtxX_d
    CHECK(cudaMemcpy(mtxX_d, mtxB_d, N * NRHS * sizeof(double), cudaMemcpyDeviceToDevice));

    // Free memory
    CHECK(cudaFree(work_d));
    CHECK(cudaFree(devIpiv));
    CHECK(cudaFree(devInfo));
}
