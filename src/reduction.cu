#include "reduction.cuh"

#include "common.hpp"
#include "cuda_utils.cuh"

#include <vector>

namespace {
constexpr int REDUCE_BLOCK = 256;

__global__ void reduceSumKernel(const float *in, float *out, int n) {
  __shared__ float sdata[REDUCE_BLOCK];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float val = 0.0f;
  if (i < static_cast<unsigned int>(n)) {
    val += in[i];
  }
  if (i + blockDim.x < static_cast<unsigned int>(n)) {
    val += in[i + blockDim.x];
  }
  sdata[tid] = val;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

float gpuReduction(const std::vector<float> &x) {
  float *dIn = nullptr;
  float *dOut = nullptr;
  int n = static_cast<int>(x.size());

  CHECK_CUDA(cudaMalloc(&dIn, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dIn, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  int currentN = n;
  float *currentIn = dIn;

  while (currentN > 1) {
    int blocks = (currentN + REDUCE_BLOCK * 2 - 1) / (REDUCE_BLOCK * 2);
    CHECK_CUDA(cudaMalloc(&dOut, blocks * sizeof(float)));

    reduceSumKernel<<<blocks, REDUCE_BLOCK>>>(currentIn, dOut, currentN);
    CHECK_CUDA(cudaGetLastError());

    if (currentIn != dIn) {
      CHECK_CUDA(cudaFree(currentIn));
    }
    currentIn = dOut;
    currentN = blocks;
  }

  float result = 0.0f;
  CHECK_CUDA(cudaMemcpy(&result, currentIn, sizeof(float), cudaMemcpyDeviceToHost));

  if (currentIn != dIn) {
    CHECK_CUDA(cudaFree(currentIn));
  }
  CHECK_CUDA(cudaFree(dIn));
  return result;
}
}  // namespace

bool runReduction() {
  constexpr int N = 1 << 20;
  std::vector<float> x(N);
  fillRandom(x);
  float cpu = cpuReductionSum(x);
  float gpu = gpuReduction(x);

  std::vector<float> ref = {cpu};
  std::vector<float> got = {gpu};
  return verify("Sum Reduction", ref, got, 1e-2f);
}
