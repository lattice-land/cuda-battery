// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cstdio>
#include <iostream>
#include <cassert>
#include <limits>

#define ON_GPU() (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#define ON_CPU() (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))

#ifdef __NVCC__
  #define CUDA_GLOBAL __global__
  #define DEVICE __device__
  #define HOST __host__
  #define CUDA DEVICE HOST

  #define CUDIE(result) { \
    cudaError_t e = (result); \
    if (e != cudaSuccess) { \
      printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }}

  #define CUDIE0() CUDIE(cudaGetLastError())
#else
  #define CUDA_GLOBAL
  #define DEVICE
  #define HOST
  #define CUDA
  #define CUDIE
  #define CUDIE0

  #include <algorithm>
  using std::min;
  using std::max;
  using std::swap;
#endif


namespace impl {
  template<typename T> CUDA void swap(T& a, T& b) {
    T c(std::move(a));
    a = std::move(b);
    b = std::move(c);
  }
}

#ifdef __NVCC__
  using impl::swap;
#endif

template<typename N>
struct Limits {
  static constexpr N bot() {
    if constexpr (std::is_floating_point<N>()) {
      return -std::numeric_limits<N>::infinity();
    }
    return std::numeric_limits<N>::min();
  }
  static constexpr N top() {
    if constexpr (std::is_floating_point<N>()) {
      return std::numeric_limits<N>::infinity();
    }
    return std::numeric_limits<N>::max();
  }
};

#ifdef DEBUG
  #define LDEBUG
  #define LOG(X) X
#else
  #define LOG(X)
#endif

#ifdef LDEBUG
  #define INFO(X) X
#else
  #define INFO(X)
#endif

#endif // UTILITY_HPP
