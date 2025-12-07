// Copyright 2021 Pierre Talbot

#ifndef CUDA_BATTERY_TUPLE_HPP
#define CUDA_BATTERY_TUPLE_HPP

// __HIP_BACKEND_NVIDIA__ replaces __NVCC__ for HIP compilation with NVIDIA backend
#ifdef __HIP_BACKEND_NVIDIA__
  // NVIDIA backend: Use CUDA-optimized tuple for better GPU performance
  #include <cuda/std/tuple>
  namespace battery {
    using cuda::std::tuple;
    using cuda::std::make_tuple;
    using cuda::std::tie;
    using cuda::std::forward_as_tuple;
    using cuda::std::tuple_cat;
    using cuda::std::get;
    using cuda::std::tuple_size;
    using cuda::std::tuple_size_v;
    using cuda::std::tuple_element;
    using cuda::std::uses_allocator;
    using cuda::std::ignore;
  }
#else
  // AMD backend or CPU: Use standard library (ROCm-optimized when available)
  #include <tuple>
  namespace battery {
    using std::tuple;
    using std::make_tuple;
    using std::tie;
    using std::forward_as_tuple;
    using std::tuple_cat;
    using std::get;
    using std::tuple_size;
    using std::tuple_size_v;
    using std::tuple_element;
    using std::uses_allocator;
    using std::ignore;
  }
#endif

#endif