#include <math.h>  // function to add the elements of two arrays

#include <iostream>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, float *x, float *y) { 
  int index = threadIdx.x; 
  int stride = blockDim.x; 
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

int main(void) {
  int N = 1 << 20;  // 1M elements
  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  // Run kernel on 1M elements on the GPU
  add<<<1, 256>>>(N, x, y);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;
  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}