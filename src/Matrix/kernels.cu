#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <class Number>
__global__ void addKernel(Number* d_m1, Number* d_m2, Number* d_m3, size_t size) {
  for (size_t i = 0; i < size; i++) {
    d_m3[i] = d_m1[i] + d_m2[i];
  }
}

__host__ void addWrapper(int* m1, int* m2, int* m3, size_t size) {

  // Pointer of arrays.
  int* d_m1;
  int* d_m2;
  int* d_m3;

  // Allocating in Device Memory.
  cudaMalloc(&d_m1, size * sizeof(int));
  cudaMalloc(&d_m2, size * sizeof(int));
  cudaMalloc(&d_m3, size * sizeof(int));

  // Copying in Device Memory.
  cudaMemcpy(d_m1, m1, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, m2, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m3, m3, size * sizeof(int), cudaMemcpyHostToDevice);

  // Calling the kernel function.
  addKernel<<<1, 1>>>(d_m1, d_m2, d_m3, size);
  cudaDeviceSynchronize();

  // Copying Device Memory to Host Memory.
  cudaMemcpy(m3, d_m3, size * sizeof(int), cudaMemcpyDeviceToHost);

  // Freeing Device Memory.
  cudaFree(d_m1);
  cudaFree(d_m2);
  cudaFree(d_m3);
}
