// Copyright 2021 Pierre Talbot

#ifndef TUPLE_HPP
#define TUPLE_HPP

#ifdef __NVCC__
  #include <cuda/std/tuple>
  using namespace cuda;
#else
  #include <tuple>
#endif

#endif