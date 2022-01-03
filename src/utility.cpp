// Copyright 2021 Pierre Talbot

#include "utility.hpp"

namespace battery {

namespace impl {
  CUDA size_t strlen(const char* str) {
    size_t n = 0;
    while(str[n] != '\0') { ++n; }
    return n;
  }
}

template<> CUDA void print(const char &x) { printf("%c", x); }
template<> CUDA void print(char const* const &x) { printf("%s", x); }
template<> CUDA void print(const int &x) { printf("%d", x); }
template<> CUDA void print(const long long int &x) { printf("%lld", x); }
template<> CUDA void print(const long int &x) { printf("%ld", x); }
template<> CUDA void print(const unsigned int &x) { printf("%u", x); }
template<> CUDA void print(const unsigned long &x) { printf("%lu", x); }
template<> CUDA void print(const unsigned long long &x) { printf("%llu", x); }
template<> CUDA void print(const float &x) { printf("%f", x); }
template<> CUDA void print(const double &x) { printf("%lf", x); }

} // namespace battery
