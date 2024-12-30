/**
 * @file cuda-tool.h
 * @author Ruinique (ruinique@foxmail.com)
 * @brief
 * @version 0.1
 * @date 2024-11-27
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <cuda_runtime_api.h>
#include <functional>

/**
 * @brief 模版实现的基于 cuda_event 的任意函数的计时器
 * @details 具体使用方法如下：
 * ```cpp
 * Ruinique_CUDA_Timer<void()> timer;
 * float time = timer.time_function([&](){
 *    // call your function here
 * });
 */
template <typename T>
class Ruinique_CUDA_Timer {
   public:
    Ruinique_CUDA_Timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~Ruinique_CUDA_Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    float time_function(T func) {
        cudaEventRecord(start, 0);
        func();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }

   private:
    cudaEvent_t start, stop;
};
