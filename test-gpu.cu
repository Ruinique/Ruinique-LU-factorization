/*
 * @Descripttion: 
 * @Author: Ruinique
 * @version: 
 * @Date: 2024-11-21 13:55:45
 * @LastEditors: Ruinique
 * @LastEditTime: 2025-01-02 15:24:32
 */
#include "common.h"
#include "lu-factorization-with-pivoting-cuslover.h"

size_t calculate_memory_usage(int64_t m, int64_t n, int64_t lda);

const int testSize[14] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 49152};

int main() {
    // 首先调用 5 轮 size = 1024 进行 warm up
    for (int i = 0; i < 5; i++) {
        float *matrix = new float[1024 * 1024];
        init_matrix_float(matrix, 1024, 1024);
        int64_t *pivot = new int64_t[1024];
        call_cusolver_to_lu_factorization_float(matrix, 1024, 1024, 1024, pivot);
        delete[] matrix;
        delete[] pivot;
    }
    float time[14];
    size_t memory[14];
    for (int i = 0; i < 14; i++) {
        int size = testSize[i];
        size_t memory_usage = calculate_memory_usage(size, size, size);
        memory[i] = memory_usage;
        float *matrix = new float[size * size];
        init_matrix_float(matrix, size, size);
        int64_t *pivot = new int64_t[size];
        time[i] = 0;
        for (int j = 0; j < 5; j++) {
            time[i] += call_cusolver_to_lu_factorization_float(matrix, size, size, size, pivot);
        }
        time[i] /= 5;
        delete[] matrix;
        delete[] pivot;
    }
    for (int i = 0; i < 14; i++) {
        printf("Time for %d x %d matrix: %f\n", testSize[i], testSize[i], time[i]);
    }
    // 将结果写入 CSV 文件
    std::ofstream csv_file("lu_factorization_times.csv");
    csv_file << "Matrix Size,Time (ms),Memory Usage (bytes)\n";
    for (int i = 0; i < 14; i++) {
        csv_file << testSize[i] << "x" << testSize[i] << "," << time[i] << "," << memory[i] << "\n";
    }
    csv_file.close();
}

size_t calculate_memory_usage(int64_t m, int64_t n, int64_t lda) {
    // 1. 矩阵 A_gpu 和 matrix_gpu 占用的内存
    size_t matrix_memory = 2 * m * n * sizeof(float); // 2块矩阵（A_gpu 和 matrix_gpu）

    // 2. 获取 device_working_buffer 和 host_working_buffer 的大小
    cusolverDnHandle_t cusolver_lu_factorization_handler;
    cusolverDnCreate(&cusolver_lu_factorization_handler);

    size_t device_working_buffer_size = 0;
    size_t host_working_buffer_size = 0;

    cusolverDnXgetrf_bufferSize(cusolver_lu_factorization_handler, nullptr, m, n, CUDA_R_32F, nullptr, lda, CUDA_R_32F,
                                &device_working_buffer_size, &host_working_buffer_size);

    cusolverDnDestroy(cusolver_lu_factorization_handler);

    // 3. pivot_gpu 占用的内存
    size_t pivot_memory = sizeof(int64_t) * std::min(m, n);

    // 4. info_lu_factorazation_gpu 占用的内存
    size_t info_memory = sizeof(int);

    // 5. 总内存使用
    size_t total_memory = matrix_memory + device_working_buffer_size + host_working_buffer_size + pivot_memory + info_memory;

    return total_memory;
}