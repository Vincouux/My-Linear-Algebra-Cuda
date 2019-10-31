template <class Number>
__host__ void addWrapper(Number* m1, Number* m2, Number* m3, size_t size) {

  // Pointer of arrays.
  Number* d_m1;
  Number* d_m2;
  Number* d_m3;

  // Allocating Device Memory.
  CudaMalloc(&d_m1, size * sizeof(Number));
  CudaMalloc(&d_m2, size * sizeof(Number));
  CudaMalloc(&d_m3, size * sizeof(Number));

  // Copying in Device Memory.
  cudaMemcpy(d_m1, m1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, m2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m3, m3, cudaMemcpyHostToDevice);

  // Calling the kernel function.
  addKernel<<<1, 1>>>(d_m1, d_m2, d_m3, size);

  // Copying in Host Memory.
  cudaMemcpy(m3, d_m3, cudaMemcpyDeviceToHost);

  // Freeing Device Memory.
  cudaFree(d_m1);
  cudaFree(d_m2);
  cudaFree(d_m3);
}

template <class Number>
__global__ void addKernel(Number* m1, Number* m2, Number* m3, size_t size) {
  for (size_t i = 0; i < size; i++) {
    m3[i] = m1[i] + m2[i];
  }
}
