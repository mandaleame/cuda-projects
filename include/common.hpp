#pragma once

#include <string>
#include <vector>

void fillRandom(std::vector<float> &v, float low = -1.0f, float high = 1.0f);
float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b);
bool verify(const std::string &name, const std::vector<float> &ref,
            const std::vector<float> &got, float tol = 1e-3f);

void cpuVectorAdd(const std::vector<float> &a, const std::vector<float> &b,
                  std::vector<float> &c);

void cpuMatMul(const std::vector<float> &A, const std::vector<float> &B,
               std::vector<float> &C, int M, int K, int N);

float cpuReductionSum(const std::vector<float> &x);

void cpuConv1D(const std::vector<float> &in, const std::vector<float> &mask,
               std::vector<float> &out);

void cpuConv2D(const std::vector<float> &in, const std::vector<float> &mask,
               std::vector<float> &out, int H, int W, int K);
