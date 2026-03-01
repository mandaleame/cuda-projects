cmake_minimum_required(VERSION 3.18)
project(cuda_algorithms LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_algorithms
  src/main.cpp
  src/common.cpp
  src/vector_add.cu
  src/matrix_mul.cu
  src/reduction.cu
  src/convolution.cu
)

target_include_directories(cuda_algorithms PRIVATE include)

target_compile_options(cuda_algorithms PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

target_link_libraries(cuda_algorithms PRIVATE CUDA::cudart CUDA::cublas)
