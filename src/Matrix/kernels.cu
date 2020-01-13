#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <typeinfo>

#include "kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        exit(code);
    }
}

template <class Number>
__global__ void addKernel(Number* d_m1, Number* d_m2, Number* d_m3, size_t size) {
    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if (index < size)
    d_m3[index] = d_m1[index] + d_m2[index];
}

template <class Number>
__host__ void addKernelWrapper(Number* m1, Number* m2, Number* m3, size_t size) {
    // Pointer of arrays.
    Number* d_m1;
    Number* d_m2;
    Number* d_m3;

    // Allocating in Device Memory.
    cudaMalloc(&d_m1, size * sizeof(Number));
    cudaMalloc(&d_m2, size * sizeof(Number));
    cudaMalloc(&d_m3, size * sizeof(Number));

    // Copying in Device Memory.
    cudaMemcpy(d_m1, m1, size * sizeof(Number), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2, size * sizeof(Number), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m3, m3, size * sizeof(Number), cudaMemcpyHostToDevice);

    // Calling the kernel function.
    size_t nbBlocks = size / 1024 + (size % 1024 ? 1 : 0);
    dim3 grid(nbBlocks), block(32, 32);
    addKernel<<<grid, block>>>(d_m1, d_m2, d_m3, size);
    cudaDeviceSynchronize();

    // Copying Device Memory to Host Memory.
    cudaMemcpy(m3, d_m3, size * sizeof(Number), cudaMemcpyDeviceToHost);

    // Freeing Device Memory.
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
}

template <class Number>
__global__ void dotKernel(Number* d_m1, Number* d_m2, Number* d_m3, size_t resultRows, size_t resultColumns, size_t interiorColumns) {
    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if (index < resultRows * resultColumns) {
        d_m3[index] = 0;
        for (int i = 0; i < interiorColumns; i++) {
            d_m3[index] += d_m1[interiorColumns * (index / resultColumns) + i] * d_m2[index % resultColumns + i * resultColumns];
        }
    }
}

template <class Number>
__host__ void dotKernelWrapper(Number* m1, Number* m2, Number* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns) {
    // Pointer of arrays.
    Number* d_m1;
    Number* d_m2;
    Number* d_m3;

    // Allocating in Device Memory.
    size_t size = resultRows * resultColumns;
    cudaMalloc(&d_m1, resultRows * interiorColumns * sizeof(Number));
    cudaMalloc(&d_m2, interiorColumns * resultColumns * sizeof(Number));
    cudaMalloc(&d_m3, size * sizeof(Number));

    // Copying in Device Memory.
    cudaMemcpy(d_m1, m1, resultRows * interiorColumns * sizeof(Number), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2, interiorColumns * resultColumns * sizeof(Number), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m3, m3, size * sizeof(Number), cudaMemcpyHostToDevice);

    // Calling the kernel function.
    size_t nbBlocks = resultRows * resultColumns / 1024 + (size % 1024 ? 1 : 0);
    dim3 grid(nbBlocks), block(32, 32);
    dotKernel<<<grid, block>>>(d_m1, d_m2, d_m3, resultRows, resultColumns, interiorColumns);
    cudaDeviceSynchronize();

    // Copying Device Memory to Host Memory.
    cudaMemcpy(m3, d_m3, size * sizeof(Number), cudaMemcpyDeviceToHost);

    // Freeing Device Memory.
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
}

template void Wrapper::add(int* m1, int* m2, int* m3, size_t size);
template void Wrapper::add(float* m1, float* m2, float* m3, size_t size);
template void Wrapper::add(double* m1, double* m2, double* m3, size_t size);
template void Wrapper::dot(int* m1, int* m2, int* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns);
template void Wrapper::dot(float* m1, float* m2, float* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns);
template void Wrapper::dot(double* m1, double* m2, double* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns);
