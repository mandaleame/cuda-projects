#include "vector_add.cuh"

#include "common.hpp"
#include "cuda_utils.cuh"

#include <algorithm>
#include <vector>

__global__ void vectorAddKernel(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

bool runVectorAdd(cublasHandle_t handle) {
  constexpr int N = 1 << 20;
  std::vector<float> a(N), b(N), cpu(N), gpuKernel(N), gpuCublas(N);
  fillRandom(a);
  fillRandom(b);
  cpuVectorAdd(a, b, cpu);

  float *ma = nullptr;
  float *mb = nullptr;
  float *mc = nullptr;

  CHECK_CUDA(cudaMallocManaged(&ma, N * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&mb, N * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&mc, N * sizeof(float)));

  std::copy(a.begin(), a.end(), ma);
  std::copy(b.begin(), b.end(), mb);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  vectorAddKernel<<<blocks, threads>>>(ma, mb, mc, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  std::copy(mc, mc + N, gpuKernel.begin());

  std::copy(b.begin(), b.end(), mc);
  const float alpha = 1.0f;
  CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, ma, 1, mc, 1));
  CHECK_CUDA(cudaDeviceSynchronize());
  std::copy(mc, mc + N, gpuCublas.begin());

  CHECK_CUDA(cudaFree(ma));
  CHECK_CUDA(cudaFree(mb));
  CHECK_CUDA(cudaFree(mc));

  bool ok1 = verify("Vector Add (Unified Memory Kernel)", cpu, gpuKernel);
  bool ok2 = verify("Vector Add (cuBLAS SAXPY)", cpu, gpuCublas);
  return ok1 && ok2;
}
