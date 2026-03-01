# CUDA Algorithms: Memory Choices and Optimization Benefits

This project implements common CUDA algorithms and validates each GPU result against a C++ CPU reference.

## Implemented Algorithms

### 1. Vector Addition

- Variants:
  - Unified Memory kernel
  - cuBLAS (`cublasSaxpy`)
- Memory used:
  - Unified Memory (`cudaMallocManaged`) for input/output buffers
  - Registers for per-thread temporary values
- Optimization benefit:
  - Unified Memory reduces code complexity and pointer management overhead.
  - cuBLAS uses heavily optimized vendor kernels and is usually faster and more robust than a custom baseline.

### 2. Matrix Multiplication

- Variants:
  - Coalesced global-memory kernel
  - Tiled shared-memory kernel
  - cuBLAS (`cublasSgemm`)
- Memory used:
  - Global memory for full matrices
  - Shared memory tiles for data reuse in the tiled kernel
  - Registers for accumulation (`sum` per thread)
- Optimization benefit:
  - Coalesced access improves effective global-memory bandwidth.
  - Shared-memory tiling reduces redundant global loads and improves arithmetic intensity.
  - cuBLAS SGEMM is highly tuned (tiling, scheduling, architecture-specific optimizations) and serves as a performance reference.

### 3. Sum Reduction

- Variant:
  - Tree-based block reduction
- Memory used:
  - Global memory for input and intermediate partial sums
  - Shared memory for intra-block reduction
  - Registers for per-thread partial accumulation
- Optimization benefit:
  - Reduces global-memory traffic by combining values in shared memory.
  - Parallel tree reduction lowers overall reduction time compared to serial accumulation.

### 4. Convolution

#### 4.1 1D Naive
- Memory used:
  - Global memory for signal, mask, and output
- Optimization benefit:
  - Simple baseline for correctness and performance comparison.

#### 4.2 1D with Constant Memory
- Memory used:
  - Constant memory for filter mask
  - Global memory for signal/output
- Optimization benefit:
  - Constant cache is efficient when many threads read the same filter coefficients, reducing effective read cost for small read-only masks.

#### 4.3 1D Tiled
- Memory used:
  - Shared memory for input tile + halo region
  - Global memory for signal/mask/output
- Optimization benefit:
  - Reuses neighboring input values across threads in a block, reducing repeated global-memory fetches.

#### 4.4 2D Naive
- Memory used:
  - Global memory for image, filter, and output
- Optimization benefit:
  - Establishes a clear stencil baseline before introducing more advanced 2D optimizations.

## Verification Approach

- Every algorithm compares GPU output against a CPU C++ reference implementation.
- Validation reports max absolute error with pass/fail tolerance.

## Build and Run

Prerequisites:
- NVIDIA GPU + compatible driver
- CUDA Toolkit (`nvcc`, CUDA runtime, cuBLAS)
- CMake 3.18+

```bash
cmake -S . -B build
cmake --build build -j
./build/cuda_algorithms
```

Note: native CUDA execution requires NVIDIA hardware/toolchain and is not supported on modern macOS systems without NVIDIA CUDA support.

## Project Layout

- `src/main.cpp`: runner for all algorithm tests
- `src/common.cpp`: CPU references + verification helpers
- `src/vector_add.cu`: vector-add kernels and checks
- `src/matrix_mul.cu`: matrix-multiply kernels and checks
- `src/reduction.cu`: sum-reduction kernel and checks
- `src/convolution.cu`: convolution kernels and checks
- `include/`: module headers and CUDA utility macros
