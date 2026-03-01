#include "convolution.cuh"

#include "common.hpp"
#include "cuda_utils.cuh"

#include <vector>

namespace {
constexpr int MAX_MASK_1D = 64;
__constant__ float kConstMask1D[MAX_MASK_1D];

__global__ void conv1DNaiveKernel(const float *in, const float *mask, float *out,
                                  int n, int maskSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  int radius = maskSize / 2;
  float sum = 0.0f;
  for (int j = -radius; j <= radius; ++j) {
    int inIdx = idx + j;
    if (inIdx >= 0 && inIdx < n) {
      sum += in[inIdx] * mask[j + radius];
    }
  }
  out[idx] = sum;
}

__global__ void conv1DConstantKernel(const float *in, float *out, int n,
                                     int maskSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  int radius = maskSize / 2;
  float sum = 0.0f;
  for (int j = -radius; j <= radius; ++j) {
    int inIdx = idx + j;
    if (inIdx >= 0 && inIdx < n) {
      sum += in[inIdx] * kConstMask1D[j + radius];
    }
  }
  out[idx] = sum;
}

__global__ void conv1DTiledKernel(const float *in, const float *mask, float *out,
                                  int n, int maskSize) {
  extern __shared__ float tile[];
  int tx = threadIdx.x;
  int globalIdx = blockIdx.x * blockDim.x + tx;
  int radius = maskSize / 2;

  int sharedCenter = tx + radius;
  tile[sharedCenter] = (globalIdx < n) ? in[globalIdx] : 0.0f;

  if (tx < radius) {
    int leftIdx = globalIdx - radius;
    tile[tx] = (leftIdx >= 0) ? in[leftIdx] : 0.0f;

    int rightIdx = globalIdx + blockDim.x;
    tile[sharedCenter + blockDim.x] = (rightIdx < n) ? in[rightIdx] : 0.0f;
  }
  __syncthreads();

  if (globalIdx < n) {
    float sum = 0.0f;
    for (int j = 0; j < maskSize; ++j) {
      sum += tile[tx + j] * mask[j];
    }
    out[globalIdx] = sum;
  }
}

__global__ void conv2DNaiveKernel(const float *in, const float *mask, float *out,
                                  int H, int W, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= H || c >= W) {
    return;
  }

  int radius = K / 2;
  float sum = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    for (int j = -radius; j <= radius; ++j) {
      int rr = r + i;
      int cc = c + j;
      if (rr >= 0 && rr < H && cc >= 0 && cc < W) {
        sum += in[rr * W + cc] * mask[(i + radius) * K + (j + radius)];
      }
    }
  }
  out[r * W + c] = sum;
}
}  // namespace

bool runConvolution() {
  constexpr int N1D = 2048;
  constexpr int MASK1D = 9;
  static_assert(MASK1D <= MAX_MASK_1D, "MASK1D exceeds constant memory mask size");

  std::vector<float> in1d(N1D), mask1d(MASK1D), outCpu1d(N1D), outNaive(N1D),
      outConst(N1D), outTiled(N1D);
  fillRandom(in1d);
  fillRandom(mask1d, -0.5f, 0.5f);
  cpuConv1D(in1d, mask1d, outCpu1d);

  float *dIn1d = nullptr;
  float *dMask1d = nullptr;
  float *dOut1d = nullptr;
  CHECK_CUDA(cudaMalloc(&dIn1d, N1D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dMask1d, MASK1D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut1d, N1D * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dIn1d, in1d.data(), N1D * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dMask1d, mask1d.data(), MASK1D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyToSymbol(kConstMask1D, mask1d.data(),
                                MASK1D * sizeof(float)));

  int threads = 256;
  int blocks = (N1D + threads - 1) / threads;

  conv1DNaiveKernel<<<blocks, threads>>>(dIn1d, dMask1d, dOut1d, N1D, MASK1D);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(outNaive.data(), dOut1d, N1D * sizeof(float),
                        cudaMemcpyDeviceToHost));

  conv1DConstantKernel<<<blocks, threads>>>(dIn1d, dOut1d, N1D, MASK1D);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(outConst.data(), dOut1d, N1D * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int radius = MASK1D / 2;
  size_t sharedBytes = (threads + 2 * radius) * sizeof(float);
  conv1DTiledKernel<<<blocks, threads, sharedBytes>>>(dIn1d, dMask1d, dOut1d, N1D,
                                                      MASK1D);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(outTiled.data(), dOut1d, N1D * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dIn1d));
  CHECK_CUDA(cudaFree(dMask1d));
  CHECK_CUDA(cudaFree(dOut1d));

  constexpr int H = 128;
  constexpr int W = 96;
  constexpr int K2D = 5;

  std::vector<float> in2d(H * W), mask2d(K2D * K2D), outCpu2d(H * W), outGpu2d(H * W);
  fillRandom(in2d);
  fillRandom(mask2d, -0.5f, 0.5f);
  cpuConv2D(in2d, mask2d, outCpu2d, H, W, K2D);

  float *dIn2d = nullptr;
  float *dMask2d = nullptr;
  float *dOut2d = nullptr;
  CHECK_CUDA(cudaMalloc(&dIn2d, in2d.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dMask2d, mask2d.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut2d, outGpu2d.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dIn2d, in2d.data(), in2d.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dMask2d, mask2d.data(), mask2d.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block2d(16, 16);
  dim3 grid2d((W + block2d.x - 1) / block2d.x, (H + block2d.y - 1) / block2d.y);
  conv2DNaiveKernel<<<grid2d, block2d>>>(dIn2d, dMask2d, dOut2d, H, W, K2D);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(outGpu2d.data(), dOut2d, outGpu2d.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dIn2d));
  CHECK_CUDA(cudaFree(dMask2d));
  CHECK_CUDA(cudaFree(dOut2d));

  bool ok1 = verify("Convolution 1D (Naive)", outCpu1d, outNaive);
  bool ok2 = verify("Convolution 1D (Constant Memory)", outCpu1d, outConst);
  bool ok3 = verify("Convolution 1D (Tiled Shared Memory)", outCpu1d, outTiled);
  bool ok4 = verify("Convolution 2D (Naive)", outCpu2d, outGpu2d);
  return ok1 && ok2 && ok3 && ok4;
}
