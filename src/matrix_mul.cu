#include "matrix_mul.cuh"

#include "common.hpp"
#include "cuda_utils.cuh"

#include <vector>

namespace {
constexpr int TILE_DIM = 16;
}

__global__ void matMulCoalescedKernel(const float *A, const float *B, float *C,
                                      int M, int K, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void matMulTiledKernel(const float *A, const float *B, float *C, int M,
                                  int K, int N) {
  __shared__ float As[TILE_DIM][TILE_DIM];
  __shared__ float Bs[TILE_DIM][TILE_DIM];

  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int col = blockIdx.x * TILE_DIM + threadIdx.x;

  float sum = 0.0f;
  int tiles = (K + TILE_DIM - 1) / TILE_DIM;
  for (int t = 0; t < tiles; ++t) {
    int aCol = t * TILE_DIM + threadIdx.x;
    int bRow = t * TILE_DIM + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_DIM; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

bool runMatrixMul(cublasHandle_t handle) {
  constexpr int M = 128;
  constexpr int K = 96;
  constexpr int N = 160;

  std::vector<float> A(M * K), B(K * N), Ccpu(M * N), Ccoalesced(M * N),
      Ctiled(M * N), Ccublas(M * N);
  fillRandom(A);
  fillRandom(B);
  cpuMatMul(A, B, Ccpu, M, K, N);

  float *dA = nullptr;
  float *dB = nullptr;
  float *dC = nullptr;

  CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, Ccoalesced.size() * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

  matMulCoalescedKernel<<<grid, block>>>(dA, dB, dC, M, K, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(Ccoalesced.data(), dC, Ccoalesced.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  matMulTiledKernel<<<grid, block>>>(dA, dB, dC, M, K, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(Ctiled.data(), dC, Ctiled.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB,
                           N, dA, K, &beta, dC, N));
  CHECK_CUDA(cudaMemcpy(Ccublas.data(), dC, Ccublas.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));

  bool ok1 = verify("MatMul (Coalesced Global Memory)", Ccpu, Ccoalesced, 2e-3f);
  bool ok2 = verify("MatMul (Cache Tiled Shared Memory)", Ccpu, Ctiled, 2e-3f);
  bool ok3 = verify("MatMul (cuBLAS SGEMM)", Ccpu, Ccublas, 2e-3f);
  return ok1 && ok2 && ok3;
}
