# 一些初始化的配置
cmake_minimum_required(VERSION 3.18.0)

set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(lu-factorazation VERSION 0.1.0 LANGUAGES C CXX CUDA)

# 创建 GPU 版本的可执行文件
add_executable(lu-factorization-gpu test-gpu.cu)

# 动态链接库
target_link_libraries(lu-factorization-gpu
    cusolver
    cublas
    cublasLt
    cusparse
)