__global__ void normalize_columns(double* mtxY_d, int numRows, int numCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < numCols) {
        // Compute the norm of the column
        double norm = 0.0;
        for (int row = 0; row < numRows; ++row) {
            double val = mtxY_d[row + col * numRows];
            norm += val * val;
        }
        norm = sqrt(norm);

        // Normalize the column
        if (norm > 0.0) {
            double inv_norm = 1.0 / norm;
            for (int row = 0; row < numRows; ++row) {
                mtxY_d[row + col * numRows] *= inv_norm;
            }
        }
    }
}
