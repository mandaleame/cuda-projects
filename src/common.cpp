#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

void fillRandom(std::vector<float> &v, float low, float high) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(low, high);
  for (auto &x : v) {
    x = dist(rng);
  }
}

float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
  float maxErr = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    maxErr = std::max(maxErr, std::fabs(a[i] - b[i]));
  }
  return maxErr;
}

bool verify(const std::string &name, const std::vector<float> &ref,
            const std::vector<float> &got, float tol) {
  float err = maxAbsDiff(ref, got);
  bool ok = err <= tol;
  std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << name
            << " | max abs error = " << err << " (tol=" << tol << ")\n";
  return ok;
}

void cpuVectorAdd(const std::vector<float> &a, const std::vector<float> &b,
                  std::vector<float> &c) {
  for (size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
}

void cpuMatMul(const std::vector<float> &A, const std::vector<float> &B,
               std::vector<float> &C, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

float cpuReductionSum(const std::vector<float> &x) {
  return std::accumulate(x.begin(), x.end(), 0.0f);
}

void cpuConv1D(const std::vector<float> &in, const std::vector<float> &mask,
               std::vector<float> &out) {
  int n = static_cast<int>(in.size());
  int m = static_cast<int>(mask.size());
  int radius = m / 2;
  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int j = -radius; j <= radius; ++j) {
      int idx = i + j;
      if (idx >= 0 && idx < n) {
        sum += in[idx] * mask[j + radius];
      }
    }
    out[i] = sum;
  }
}

void cpuConv2D(const std::vector<float> &in, const std::vector<float> &mask,
               std::vector<float> &out, int H, int W, int K) {
  int radius = K / 2;
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
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
  }
}
