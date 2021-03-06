// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cstdio>
#include <cassert>
#include <limits>
#include <climits>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cfenv>
#include <bit>

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
#endif

namespace battery {
namespace impl {
  template<class T> CUDA inline void swap(T& a, T& b) {
    T c(std::move(a));
    a = std::move(b);
    b = std::move(c);
  }

  CUDA inline size_t strlen(const char* str) {
    size_t n = 0;
    while(str[n] != '\0') { ++n; }
    return n;
  }

  /** See https://stackoverflow.com/a/34873406/2231159 */
  CUDA inline int strcmp(const char* s1, const char* s2) {
    while(*s1 && (*s1 == *s2)) {
      s1++;
      s2++;
    }
    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
  }
}

template<class T> CUDA inline void swap(T& a, T& b) {
  #ifdef __CUDA_ARCH__
    impl::swap(a, b);
  #else
    std::swap(a, b);
  #endif
}

CUDA inline size_t strlen(const char* str) {
  #ifdef __CUDA_ARCH__
    return impl::strlen(str);
  #else
    return std::strlen(str);
  #endif
}

/** See https://stackoverflow.com/a/34873406/2231159 */
CUDA inline int strcmp(const char* s1, const char* s2) {
  #ifdef __CUDA_ARCH__
    return impl::strcmp(s1, s2);
  #else
    return std::strcmp(s1, s2);
  #endif
}

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

template<class T> CUDA T isnan(T a) {
  #ifdef __CUDA_ARCH__
    return ::isnan(a);
  #else
    return std::isnan(a);
  #endif
}

CUDA inline float nextafter(float f, float dir) {
  #ifdef __CUDA_ARCH__
    return ::nextafterf(f, dir);
  #else
    return std::nextafterf(f, dir);
  #endif
}

CUDA inline double nextafter(double f, double dir) {
  #ifdef __CUDA_ARCH__
    return ::nextafter(f, dir);
  #else
    return std::nextafter(f, dir);
  #endif
}

template<class T>
struct Limits {
  static constexpr T bot() {
    if constexpr (std::is_floating_point<T>()) {
      return -std::numeric_limits<T>::infinity();
    }
    return std::numeric_limits<T>::min();
  }
  static constexpr T top() {
    if constexpr (std::is_floating_point<T>()) {
      return std::numeric_limits<T>::infinity();
    }
    return std::numeric_limits<T>::max();
  }
};

#define MAP_LIMITS(x, From, To) \
  if(x == 0) { return 0; } \
  if(x == Limits<From>::bot()) {\
    return Limits<To>::bot();   \
  }                             \
  if(x == Limits<From>::top()) {\
    return Limits<To>::top();   \
  }

/** Cast the variable `x` from type `From` to type `To` following upper rounding rule (cast in the direction of infinity).
  Minimal and maximal values of `From` are interpreted as infinities, and are therefore mapped to the infinities of the new types accordingly (e.g., float INF maps to int MAX_INT).

  * On CPU: Rounding mode is UPWARD after this operation.
  * On GPU: CUDA intrinsics are used.

  Overflow: Nothing is done to prevent overflow, it mostly behaves as with `static_cast`. */
template<class To, class From>
CUDA To ru_cast(From x) {
  if constexpr(std::is_same_v<To, From>) {
    return x;
  }
  MAP_LIMITS(x, From, To)
  #ifdef __CUDA_ARCH__
    // Integer to floating-point number cast.
    if constexpr(std::is_integral_v<From> && std::is_floating_point_v<To>) {
      if constexpr(std::is_same_v<From, unsigned long long>) {
        if constexpr(std::is_same_v<To, float>) {
          return __ull2float_ru(x);
        }
        else if constexpr(std::is_same_v<To, double>) {
          return __ull2double_ru(x);
        }
        else {
          static_assert(std::is_same_v<To, float>, "Unsupported combination of types in ru_cast.");
        }
      }
      else if constexpr(std::is_same_v<From, int>) {
        if constexpr(std::is_same_v<To, float>) {
          return __int2float_ru(x);
        }
        else if constexpr(std::is_same_v<To, double>) {
          return __int2double_rn(x);
        }
        else {
          static_assert(std::is_same_v<To, float>, "Unsupported combination of types in ru_cast.");
        }
      }
      else {
        static_assert(sizeof(long long int) >= sizeof(From));
        if constexpr(std::is_same_v<To, float>) {
          return __ll2float_ru(x);
        }
        else if constexpr(std::is_same_v<To, double>) {
          return __ll2double_ru(x);
        }
        else {
          static_assert(std::is_same_v<To, float>, "Unsupported combination of types in ru_cast.");
        }
      }
    }
    // Floating-point number to integer number.
    else if constexpr(std::is_floating_point_v<From> && std::is_integral_v<To>) {
      if constexpr(std::is_same_v<From, float>) {
        return static_cast<To>(__float2ll_ru(x));
      }
      else if constexpr(std::is_same_v<From, double>) {
        return static_cast<To>(__double2ll_ru(x));
      }
      else {
        static_assert(std::is_same_v<From, float>, "Unsupported combination of types in ru_cast.");
      }
    }
    // Floating-point to floating-point.
    else if constexpr(std::is_same_v<From, double> && std::is_same_v<To, float>) {
      return __double2float_ru(x);
    }
  #else
    // Integer to floating-point number cast.
    if constexpr(std::is_integral_v<From> && std::is_floating_point_v<To>) {
      int r = std::fesetround(FE_UPWARD);
      assert(r == 0);
      return static_cast<To>(x);
    }
    // Floating-point number to integer number.
    else if constexpr(std::is_floating_point_v<From> && std::is_integral_v<To>) {
      return static_cast<To>(std::ceil(x));
    }
    // Floating-point to floating-point.
    else if constexpr(std::is_same_v<From, double> && std::is_same_v<To, float>) {
      int r = std::fesetround(FE_UPWARD);
      assert(r == 0);
      return static_cast<To>(x);
    }
  #endif
  return static_cast<To>(x);
}

/** Cast the variable `x` from type `From` to type `To` following down rounding rule (cast in the direction of negative infinity).
  Minimal and maximal values of `From` are interpreted as infinities, and are therefore mapped to the infinities of the new types accordingly (e.g., float INF maps to int MAX_INT).

  * On CPU: Rounding mode is DOWNWARD after this operation.
  * On GPU: CUDA intrinsics are used.

  Overflow: Nothing is done to prevent overflow, it mostly behaves as with `static_cast`. */
template<class To, class From>
CUDA To rd_cast(From x) {
  if constexpr(std::is_same_v<To, From>) {
    return x;
  }
  MAP_LIMITS(x, From, To)
  #ifdef __CUDA_ARCH__
    // Integer to floating-point number cast.
    if constexpr(std::is_integral_v<From> && std::is_floating_point_v<To>) {
      if constexpr(std::is_same_v<From, unsigned long long>) {
        if constexpr(std::is_same_v<To, float>) {
          return __ull2float_rd(x);
        }
        else if constexpr(std::is_same_v<To, double>) {
          return __ull2double_rd(x);
        }
        else {
          static_assert(std::is_same_v<To, float>, "Unsupported combination of types in rd_cast.");
        }
      }
      else if constexpr(std::is_same_v<From, int>) {
        if constexpr(std::is_same_v<To, float>) {
          return __int2float_rd(x);
        }
        else if constexpr(std::is_same_v<To, double>) {
          return __int2double_rn(x);
        }
        else {
          static_assert(std::is_same_v<To, float>, "Unsupported combination of types in rd_cast.");
        }
      }
      else {
        static_assert(sizeof(long long int) >= sizeof(From));
        if constexpr(std::is_same_v<To, float>) {
          return __ll2float_rd(x);
        }
        else if constexpr(std::is_same_v<To, double>) {
          return __ll2double_rd(x);
        }
        else {
          static_assert(std::is_same_v<To, float>, "Unsupported combination of types in rd_cast.");
        }
      }
    }
    // Floating-point number to integer number.
    else if constexpr(std::is_floating_point_v<From> && std::is_integral_v<To>) {
      if constexpr(std::is_same_v<From, float>) {
        return static_cast<To>(__float2ll_rd(x));
      }
      else if constexpr(std::is_same_v<From, double>) {
        return static_cast<To>(__double2ll_rd(x));
      }
      else {
        static_assert(std::is_same_v<To, float>, "Unsupported combination of types in rd_cast.");
      }
    }
    // Floating-point to floating-point.
    else if constexpr(std::is_same_v<From, double> && std::is_same_v<To, float>) {
      return __double2float_rd(x);
    }
  #else
    // Integer to floating-point number cast.
    if constexpr(std::is_integral_v<From> && std::is_floating_point_v<To>) {
      int r = std::fesetround(FE_DOWNWARD);
      assert(r == 0);
      return static_cast<To>(x);
    }
    // Floating-point number to integer number.
    else if constexpr(std::is_floating_point_v<From> && std::is_integral_v<To>) {
      return static_cast<To>(std::floor(x));
    }
    // Floating-point to floating-point.
    else if constexpr(std::is_same_v<From, double> && std::is_same_v<To, float>) {
      int r = std::fesetround(FE_DOWNWARD);
      assert(r == 0);
      return static_cast<To>(x);
    }
  #endif
  return static_cast<To>(x);
}

template<class T>
CUDA int popcount(T x) {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "popcount only works on unsigned integers");
  #ifdef __CUDA_ARCH__
    if constexpr(std::is_same_v<T, unsigned int>) {
      return __popc(x);
    }
    else if constexpr(std::is_same_v<T, unsigned long long>) {
      return __popcll(x);
    }
    else {
      return __popcll(static_cast<unsigned long long>(x));
    }
  #elif __cpp_lib_bitops
    return std::popcount(x);
  #else
    int c = 0;
    for(int i = 0; i < sizeof(T) * CHAR_BIT && x != 0; ++i) {
      c += (x & 1);
      x >>= 1;
    }
    return c;
  #endif
}

template<class T>
CUDA int countl_zero(T x) {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "countl_zero only works on unsigned integers");
  #ifdef __CUDA_ARCH__
    // If the size of `T` is smaller than `int` or `long long int` we must remove the extra zeroes that are added after conversion.
    if constexpr(sizeof(T) <= sizeof(int)) {
      return __clz(x) - ((sizeof(int) - sizeof(T)) * CHAR_BIT);
    }
    else if constexpr(sizeof(T) <= sizeof(long long int)) {
      return __clzll(x) - ((sizeof(long long int) - sizeof(T)) * CHAR_BIT);
    }
    else {
      static_assert(sizeof(T) < sizeof(long long int), "countX_Y (CUDA version) only supports types smaller than long long int.");
    }
  #elif __cpp_lib_bitops
    return std::countl_zero(x);
  #else
    int c = 0;
    constexpr int bits = sizeof(T) * CHAR_BIT;
    constexpr T mask = (T)1 << (bits - 1);
    for(int i = 0; i < bits && (x & mask) == 0; ++i) {
      c += (x & mask) == 0;
      x <<= 1;
    }
    return c;
  #endif
}

template<class T>
CUDA int countl_one(T x) {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "countl_one only works on unsigned integers");
  #ifdef __CUDA_ARCH__
    return countl_zero((T)~x);
  #elif __cpp_lib_bitops
    return std::countl_one(x);
  #else
    int c = 0;
    constexpr int bits = sizeof(T) * CHAR_BIT;
    constexpr T mask = (T)1 << (bits - 1);
    for(int i = 0; i < bits && (x & mask) > 0; ++i) {
      c += (x & mask) > 0;
      x <<= 1;
    }
    return c;
  #endif
}

template<class T>
CUDA int countr_zero(T x) {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "countl_zero only works on unsigned integers");
  #ifdef __CUDA_ARCH__
    if(x == 0) {
      return sizeof(T) * CHAR_BIT;
    }
    if constexpr(sizeof(T) <= sizeof(int)) {
      return __ffs(x) - 1;
    }
    else if constexpr(sizeof(T) <= sizeof(long long int)) {
      return __ffsll(x) - 1;
    }
    else {
      static_assert(sizeof(T) < sizeof(long long int), "countr_zero (CUDA version) only supports types smaller or equal to long long int.");
    }
  #elif __cpp_lib_bitops
    return std::countr_zero(x);
  #else
    int c = 0;
    constexpr int bits = sizeof(T) * CHAR_BIT;
    constexpr T mask = 1;
    for(int i = 0; i < bits && (x & mask) == 0; ++i) {
      c += (x & mask) == 0;
      x >>= 1;
    }
    return c;
  #endif
}

template<class T>
CUDA int countr_one(T x) {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "countr_one only works on unsigned integers");
  #ifdef __CUDA_ARCH__
    return countr_zero((T)~x);
  #elif __cpp_lib_bitops
    return std::countr_one(x);
  #else
    int c = 0;
    constexpr int bits = sizeof(T) * CHAR_BIT;
    constexpr T mask = 1;
    for(int i = 0; i < bits && (x & mask) > 0; ++i) {
      c += (x & mask) > 0;
      x >>= 1;
    }
    return c;
  #endif
}

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
template<> CUDA inline void print(const short &x) { printf("%d", (int)x); }
template<> CUDA inline void print(const int &x) { printf("%d", x); }
template<> CUDA inline void print(const long long int &x) { printf("%lld", x); }
template<> CUDA inline void print(const long int &x) { printf("%ld", x); }
template<> CUDA inline void print(const unsigned char &x) { printf("%d", (int)x); }
template<> CUDA inline void print(const unsigned short &x) { printf("%d", (int)x); }
template<> CUDA inline void print(const unsigned int &x) { printf("%u", x); }
template<> CUDA inline void print(const unsigned long &x) { printf("%lu", x); }
template<> CUDA inline void print(const unsigned long long &x) { printf("%llu", x); }
template<> CUDA inline void print(const float &x) { printf("%f", x); }
template<> CUDA inline void print(const double &x) { printf("%lf", x); }
template<> CUDA inline void print(char const* const &x) { printf("%s", x); }

} // namespace battery

#endif // UTILITY_HPP
