// Copyright 2021 Pierre Talbot

#ifndef CUDA_BATTERY_TUPLE_HPP
#define CUDA_BATTERY_TUPLE_HPP

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
  using cuda::std::tuple_size_v;
  using cuda::std::tuple_element;
  using cuda::std::uses_allocator;
  using cuda::std::ignore;
#else
#ifndef __HIP_DEVICE_COMPILE__
  using std::tuple;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::make_tuple;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::tie;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::forward_as_tuple;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::tuple_cat;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::get;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::tuple_size;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::tuple_size_v;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::tuple_element;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::uses_allocator;
#endif
#ifndef __HIP_DEVICE_COMPILE__
  using std::ignore;
#endif
#endif
}

#endif