void normalize_Den_Mtx(double* mtxY_d, int numRows, int numCols) {
    int blockSize = 256;
    int gridSize = (numCols + blockSize - 1) / blockSize;

    normalize_columns<<<gridSize, blockSize>>>(mtxY_d, numRows, numCols);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}
