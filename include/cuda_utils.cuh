#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                         \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "   \
                << cudaGetErrorString(err) << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CHECK_CUBLAS(call)                                                       \
  do {                                                                           \
    cublasStatus_t st = (call);                                                  \
    if (st != CUBLAS_STATUS_SUCCESS) {                                           \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " -> "  \
                << st << std::endl;                                              \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)
