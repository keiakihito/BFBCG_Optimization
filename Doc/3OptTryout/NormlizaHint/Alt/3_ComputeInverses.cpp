__global__ void compute_inverses(double* norms, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (norms[idx] > 0.0) {
            norms[idx] = 1.0 / norms[idx];
        } else {
            norms[idx] = 0.0;
        }
    }
}

// Launch kernel
int blockSize = 256;
int gridSize = (currentRank + blockSize - 1) / blockSize;
compute_inverses<<<gridSize, blockSize>>>(norms_d, currentRank);
