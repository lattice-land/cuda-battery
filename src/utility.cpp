// Copyright 2021 Pierre Talbot

#include "utility.hpp"

namespace impl {
  CUDA size_t strlen(const char* str) {
    size_t n = 0;
    while(str[n] != '\0') { ++n; }
    return n;
  }
}
