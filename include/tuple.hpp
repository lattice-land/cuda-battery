// Copyright 2021 Pierre Talbot

#ifndef TUPLE_HPP
#define TUPLE_HPP

#ifdef __NVCC__
  #include <cuda/std/tuple>
#else
  #include <tuple>
#endif

namespace battery {
#ifdef __NVCC__
  using cuda::std::tuple;
  using cuda::std::make_tuple;
  using cuda::std::tie;
  using cuda::std::forward_as_tuple;
  using cuda::std::tuple_cat;
  using cuda::std::get;
  using cuda::std::tuple_size;
  using cuda::std::tuple_element;
  using cuda::std::uses_allocator;
  using cuda::std::ignore;
#else
  using std::tuple;
  using std::make_tuple;
  using std::tie;
  using std::forward_as_tuple;
  using std::tuple_cat;
  using std::get;
  using std::tuple_size;
  using std::tuple_element;
  using std::uses_allocator;
  using std::ignore;
#endif
}

#endif