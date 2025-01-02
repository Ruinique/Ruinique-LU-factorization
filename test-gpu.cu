/*
 * @Descripttion: 
 * @Author: Ruinique
 * @version: 
 * @Date: 2024-11-21 13:55:45
 * @LastEditors: Ruinique
 * @LastEditTime: 2025-01-02 14:18:34
 */
#include "common.h"
#include "lu-factorization-with-pivoting-cuslover.h"

const int testSize[12] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

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
    float time[12];
    for (int i = 0; i < 12; i++) {
        int size = testSize[i];
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
    for (int i = 0; i < 12; i++) {
        printf("Time for %d x %d matrix: %f\n", testSize[i], testSize[i], time[i]);
    }
    // 将结果写入 CSV 文件
    std::ofstream csv_file("lu_factorization_times.csv");
    csv_file << "Matrix Size,Time (ms)\n";
    for (int i = 0; i < 12; i++) {
        csv_file << testSize[i] << "x" << testSize[i] << "," << time[i] << "\n";
    }
    csv_file.close();
}