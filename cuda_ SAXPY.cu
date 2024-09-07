#include <stdio.h>

// CUDA kernel function to perform SAXPY (Single-precision A·X Plus Y)
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        y[i] = a * x[i] + y[i];
}

int main(void) {
    int N = 1 << 20; // Set N to 2^20 (approximately 1 million)
    
    float *x, *y, *d_x, *d_y;

    // Allocate memory on the host (CPU)
    x = (float*) malloc(N * sizeof(float));
    y = (float*) malloc(N * sizeof(float));

    // Allocate memory on the device (GPU)
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Initialize host data
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f; // Initialize x with 1.0
        y[i] = 2.0f; // Initialize y with 2.0
    }

    // Copy host data to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on N (1M) elements
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 4.0f)); // Expected result: y[i] = 4.0f
    }
    printf("Max error: %f\n", maxError);

    // Free GPU and CPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}
