/*
 * @Descripttion: 
 * @Author: Ruinique
 * @version: 
 * @Date: 2024-11-21 13:47:46
 * @LastEditors: Ruinique
 * @LastEditTime: 2024-12-30 16:45:47
 */
/**
 * @file lu-factorazation-with-pivoting-cuslover.h
 * @author ruinique (ruin1que@outlook.com)
 * @brief 一个调用 cusolver 来实现选主元的 LU 分解的函数
 * @date 2024-11-21
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <cusolverDn.h>

#include "common.h"
#include "cuda-tool.h"

void checkInfo4getrf(int *info_lu_factorazation_cpu);

void checkAns4Cusolver(double *matrix, double *matrix_gpu, int64_t &m,
                       int64_t &n, int64_t *pivot, int64_t *pivot_gpu);

float call_cusolver_to_lu_factorization_double(double *matrix, int64_t m,
                                              int64_t n, int64_t lda,
                                              int64_t *pivot) {
    double *A_gpu;
    cudaMalloc((void **)&A_gpu, m * n * sizeof(double));
    cudaMemcpy(A_gpu, matrix, m * n * sizeof(double), cudaMemcpyHostToDevice);
    // create cusolver handle
    cusolverDnHandle_t cusolver_lu_factorization_handler;
    cusolverDnCreate(&cusolver_lu_factorization_handler);
    // get size of working buffer
    size_t device_working_buffer_size;
    size_t host_working_buffer_size;
    cusolverDnXgetrf_bufferSize(cusolver_lu_factorization_handler, nullptr, m,
                                n, CUDA_R_64F, matrix, lda, CUDA_R_64F,
                                &device_working_buffer_size,
                                &host_working_buffer_size);    
    // allocate related resource
    int *info_lu_factorazation_gpu;
    cudaMalloc((void **)&info_lu_factorazation_gpu, sizeof(int));
    void *host_working_buffer = new char[host_working_buffer_size];
    void *device_working_buffer;
    cudaMalloc(&device_working_buffer, device_working_buffer_size);
    double *matrix_gpu;
    cudaMalloc((void **)&matrix_gpu, m * n * sizeof(double));
    cudaMemcpy(matrix_gpu, matrix, m * n * sizeof(double),
               cudaMemcpyHostToDevice);
    int64_t *pivot_gpu;
    cudaMalloc((void **)&pivot_gpu, std::min(m, n) * sizeof(int64_t));
    Ruinique_CUDA_Timer<std::function<void()>> timer;
    float time = timer.time_function([&]() {
        // call cusolver
        cusolverDnXgetrf(cusolver_lu_factorization_handler, nullptr, m, n,
                         CUDA_R_64F, matrix_gpu, lda, pivot_gpu, CUDA_R_64F,
                         device_working_buffer, device_working_buffer_size,
                         host_working_buffer, host_working_buffer_size,
                         info_lu_factorazation_gpu);
    });
    printf("cusolverDnXgetrf time: %f ms\n", time);
    // check res
    int info_lu_factorazation_cpu = 0;
    cudaMemcpy(&info_lu_factorazation_cpu, info_lu_factorazation_gpu,
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(info_lu_factorazation_gpu);
    if (info_lu_factorazation_cpu != 0) {
        throw std::runtime_error("LU factorization failed with non-zero info.");
    }
    cudaMemcpy(matrix, matrix_gpu, m * n * sizeof(double),
               cudaMemcpyDeviceToHost);
    // transport res to host
    // checkAns4Cusolver(matrix, matrix_gpu, m, n, pivot, pivot_gpu);
    // free resource
    cudaFree(matrix_gpu);
    cudaFree(pivot_gpu);
    delete[] (char *)host_working_buffer;
    cudaFree(device_working_buffer);
    cudaFree(A_gpu);
    cusolverDnDestroy(cusolver_lu_factorization_handler);
    // return time for test
    return time;
}

void checkAns4Cusolver(double *matrix, double *matrix_gpu, int64_t &m,
                       int64_t &n, int64_t *pivot, int64_t *pivot_gpu) {
    print_matrix_double(matrix, m, n);
    cudaMemcpy(pivot, pivot_gpu, std::min(m, n) * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    print_matrix_int64(build_p_matrix_by_ipiv(pivot, m), m, m);
}