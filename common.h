/*
 * @Descripttion: 
 * @Author: Ruinique
 * @version: 
 * @Date: 2024-11-21 13:01:20
 * @LastEditors: Ruinique
 * @LastEditTime: 2025-01-02 14:18:04
 */
/**
 * @file common.h
 * @author ruinique (ruinique@foxmail.com)
 * @brief 一些公共函数
 * @date 2024-11-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once


#include <algorithm>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <fstream>

void init_matrix(double *A, int m, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[j * m + i] = rand() % 10;
        }
    }
}

void init_matrix_float(float *A, size_t m, size_t n) {
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++) {
            A[j * m + i] = rand() % 10;
        }
    }
}

void print_matrix_double(double *A, int m, int n) {
    printf("Matrix %d x %d:\n", m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f ", A[j * m + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_int(int *A, int m, int n) {
    printf("Matrix %d x %d:\n", m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", A[j * m + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_int64(int64_t *A, int m, int n) {
    printf("Matrix %d x %d:\n", m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%ld ", A[j * m + i]);
        }
        printf("\n");
    }
    printf("\n");
}


int64_t* build_p_matrix_by_ipiv(int64_t *pivot, int m) {
    int64_t *P = new int64_t[m * m];
    // 初始矩阵 P 为单位矩阵，并且列优先存储
    for (int j = 0;j < m;j++) {
        for (int i = 0;i < m;i++) {
            P[j * m + i] = i == j ? 1 : 0;
        }
    }
    // 根据 pivot 重新排列 P, 其中 pivot 是从 1 开始的
    for (int i = 0;i < m;i++) {
        int64_t pivot_i = pivot[i] - 1;
        if (pivot_i != i) {
            for (int j = 0;j < m;j++) {
                int64_t tmp = P[j * m + i];
                P[j * m + i] = P[j * m + pivot_i];
                P[j * m + pivot_i] = tmp;
            }
        }
    }
    return P;
}

void build_l_u_matrix_by_res(double *matrix, int m, int n, double *L, double *U) {
    for (int i = 0;i < m;i++) {
        for (int j = 0;j < n;j++) {
            if (i > j) {
                L[j * m + i] = matrix[j * m + i];
            } else if (i == j) {
                L[j * m + i] = 1;
                U[j * m + i] = matrix[j * m + i];
            } else {
                U[j * m + i] = matrix[j * m + i];
            }
        }
    }
}