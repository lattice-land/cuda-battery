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
  template<class T> CUDA constexpr inline void swap(T& a, T& b) {
    T c(std::move(a));
    a = std::move(b);
    b = std::move(c);
  }

  CUDA constexpr inline size_t strlen(const char* str) {
    size_t n = 0;
    while(str[n] != '\0') { ++n; }
    return n;
  }

  /** See https://stackoverflow.com/a/34873406/2231159 */
  CUDA constexpr inline int strcmp(const char* s1, const char* s2) {
    while(*s1 && (*s1 == *s2)) {
      s1++;
      s2++;
    }
    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
  }
}

template<class T> CUDA constexpr inline void swap(T& a, T& b) {
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

template<class T> CUDA constexpr T min(T a, T b) {
  #ifdef __CUDA_ARCH__
    // When C++23 is available
    // if !consteval { return ::min(a, b); }
    // else { return std::min(a, b); }
    return a < b ? a : b;
  #else
    return std::min(a, b);
  #endif
}

template<class T> CUDA constexpr T max(T a, T b) {
  #ifdef __CUDA_ARCH__
    // When C++23 is available
    // if !consteval { return ::max(a, b); }
    // else { return std::max(a, b); }
    return a > b ? a : b;
  #else
    return std::max(a, b);
  #endif
}

template<class T> CUDA constexpr T isnan(T a) {
  #ifdef __CUDA_ARCH__
    return ::isnan(a);
  #else
    return std::isnan(a);
  #endif
}

CUDA constexpr inline float nextafter(float f, float dir) {
  #ifdef __CUDA_ARCH__
    return ::nextafterf(f, dir);
  #else
    return std::nextafterf(f, dir);
  #endif
}

CUDA constexpr inline double nextafter(double f, double dir) {
  #ifdef __CUDA_ARCH__
    return ::nextafter(f, dir);
  #else
    return std::nextafter(f, dir);
  #endif
}

template<class T>
struct limits {
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
  if(x == limits<From>::bot()) {\
    return limits<To>::bot();   \
  }                             \
  if(x == limits<From>::top()) {\
    return limits<To>::top();   \
  }

/** Cast the variable `x` from type `From` to type `To` following upper rounding rule (cast in the direction of infinity).
  Minimal and maximal values of `From` are interpreted as infinities, and are therefore mapped to the infinities of the new types accordingly (e.g., float INF maps to int MAX_INT).

  * On CPU: Rounding mode is UPWARD after this operation.
  * On GPU: CUDA intrinsics are used.

  Overflow: Nothing is done to prevent overflow, it mostly behaves as with `static_cast`. */
template<class To, class From, bool map_limits = true>
CUDA constexpr To ru_cast(From x) {
  if constexpr(std::is_same_v<To, From>) {
    return x;
  }
  if constexpr(map_limits) {
    MAP_LIMITS(x, From, To)
  }
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
      #ifndef __GNUC__
        #pragma STDC FENV_ACCESS ON
      #endif
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
      #ifndef __GNUC__
        #pragma STDC FENV_ACCESS ON
      #endif
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
template<class To, class From, bool map_limits=true>
CUDA constexpr To rd_cast(From x) {
  if constexpr(std::is_same_v<To, From>) {
    return x;
  }
  if constexpr(map_limits) {
    MAP_LIMITS(x, From, To)
  }
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
      #ifndef __GNUC__
        #pragma STDC FENV_ACCESS ON
      #endif
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
      #ifndef __GNUC__
        #pragma STDC FENV_ACCESS ON
      #endif
      int r = std::fesetround(FE_DOWNWARD);
      assert(r == 0);
      return static_cast<To>(x);
    }
  #endif
  return static_cast<To>(x);
}

template<class T>
CUDA constexpr int popcount(T x) {
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
CUDA constexpr int countl_zero(T x) {
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
CUDA constexpr int countl_one(T x) {
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
CUDA constexpr int countr_zero(T x) {
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
CUDA constexpr int countr_one(T x) {
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

/** Signum function, https://stackoverflow.com/a/4609795/2231159 */
template <class T>
CUDA constexpr int signum(T val) {
  return (T(0) < val) - (val < T(0));
}

/** Precondition: T is an integer with b >= 0.*/
template <class T>
CUDA constexpr T ipow(T a, T b) {
  static_assert(std::is_integral_v<T>, "ipow is only working on integer value.");
  assert(b >= 0);
  if(b == 2) {
    return a*a;
  }
  // Code taken from GeCode implementation.
  T p = 1;
  do {
    if (b % 2 == 0) {
      a *= a;
      b >>= 1;
    } else {
      p *= a;
      b--;
    }
  } while (b > 0);
  return p;
}

#define FLOAT_ARITHMETIC_CUDA_IMPL(name, cudaname)   \
  if constexpr(std::is_same_v<T, float>) {           \
    return __f ## cudaname(x, y);                    \
  }                                                  \
  else if constexpr(std::is_same_v<T, double>) {     \
    return __d ## cudaname(x, y);                    \
  }                                                  \
  else {                                             \
    static_assert(std::is_same_v<T, float>, #name " (CUDA version) only support float or double types."); \
  }

#define FLOAT_ARITHMETIC_CPP_IMPL(cppop, cppround) \
  int r = std::fesetround(cppround);              \
  assert(r == 0);                                  \
  return x cppop y;

template <class T>
CUDA constexpr T add_up(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(add_up, add_ru)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(+, FE_UPWARD)
  #endif
}

template <class T>
CUDA constexpr T add_down(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(add_down, add_rd)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(+, FE_DOWNWARD)
  #endif
}

template <class T>
CUDA constexpr T sub_up(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(sub_up, sub_ru)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(-, FE_UPWARD)
  #endif
}

template <class T>
CUDA constexpr T sub_down(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(sub_down, sub_rd)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(-, FE_DOWNWARD)
  #endif
}

template <class T>
CUDA constexpr T mul_up(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(mul_up, mul_ru)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(*, FE_UPWARD)
  #endif
}

template <class T>
CUDA constexpr T mul_down(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(mul_down, mul_rd)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(*, FE_DOWNWARD)
  #endif
}

template <class T>
CUDA constexpr T div_up(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(div_up, div_ru)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(/, FE_UPWARD)
  #endif
}

template <class T>
CUDA constexpr T div_down(T x, T y) {
  #ifdef __CUDA_ARCH__
    FLOAT_ARITHMETIC_CUDA_IMPL(div_down, div_rd)
  #else
    #ifndef __GNUC__
      #pragma STDC FENV_ACCESS ON
    #endif
    FLOAT_ARITHMETIC_CPP_IMPL(/, FE_DOWNWARD)
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
