#include "convolution.cuh"
#include "cuda_utils.cuh"
#include "matrix_mul.cuh"
#include "reduction.cuh"
#include "vector_add.cuh"

#include <cstdlib>
#include <iostream>

int main() {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  bool allOk = true;
  allOk = runVectorAdd(handle) && allOk;
  allOk = runMatrixMul(handle) && allOk;
  allOk = runReduction() && allOk;
  allOk = runConvolution() && allOk;

  CHECK_CUBLAS(cublasDestroy(handle));

  std::cout << "\nOverall result: " << (allOk ? "ALL TESTS PASSED" : "FAILURES FOUND")
            << std::endl;
  return allOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
