#pragma once

#include <chrono>
#include <functional>
#include <iostream>

#include <cuda_runtime.h>

static void timeit(int times, const std::function<void()> &f) {
  cudaDeviceSynchronize();
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int _ = 0; _ < times; ++_) {
    f();
  }
  cudaDeviceSynchronize();
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << "Elapsed "
            << std::chrono::duration_cast<std::chrono::microseconds>(t_end -
                                                                     t_start)
                       .count() /
                   1000.0 / times
            << "ms" << std::endl;
}
