// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cstdio>
#include <cassert>
#include <limits>

#ifdef __NVCC__
  #define CUDA_GLOBAL __global__
  #define DEVICE __device__
  #define HOST __host__
  #define SHARED __shared__
  #define CUDA DEVICE HOST
  #define INLINE __forceinline__

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
  #define SHARED
  #define CUDA
  #define CUDIE(S) S
  #define CUDIE0
  #define INLINE inline

  #include <algorithm>
  #include <cstring> // for strlen

  namespace battery {
    using std::swap;
    using std::strlen;
  }
#endif

namespace battery {
namespace impl {
  template<class T> CUDA void swap(T& a, T& b) {
    T c(std::move(a));
    a = std::move(b);
    b = std::move(c);
  }

  CUDA inline size_t strlen(const char* str) {
    size_t n = 0;
    while(str[n] != '\0') { ++n; }
    return n;
  }
}

#ifdef __NVCC__
  using impl::swap;
  using impl::strlen;
#endif

template<class T> CUDA T min(T a, T b) {
  #ifdef __CUDA_ARCH__
    return ::min(a, b);
  #else
    return std::min(a, b);
  #endif
}

template<class T> CUDA T max(T a, T b) {
  #ifdef __CUDA_ARCH__
    return ::max(a, b);
  #else
    return std::max(a, b);
  #endif
}

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

template<typename T>
CUDA inline void print(const T& t) {
  t.print();
}
template<> CUDA inline void print(const char &x) { printf("%c", x); }
template<> CUDA inline void print(char const* const &x) { printf("%s", x); }
template<> CUDA inline void print(const int &x) { printf("%d", x); }
template<> CUDA inline void print(const long long int &x) { printf("%lld", x); }
template<> CUDA inline void print(const long int &x) { printf("%ld", x); }
template<> CUDA inline void print(const unsigned int &x) { printf("%u", x); }
template<> CUDA inline void print(const unsigned long &x) { printf("%lu", x); }
template<> CUDA inline void print(const unsigned long long &x) { printf("%llu", x); }
template<> CUDA inline void print(const float &x) { printf("%f", x); }
template<> CUDA inline void print(const double &x) { printf("%lf", x); }

} // namespace battery

#endif // UTILITY_HPP
