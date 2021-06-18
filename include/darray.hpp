// Copyright 2021 Pierre Talbot

#ifndef ARRAY_HPP
#define ARRAY_HPP

#include "utility.hpp"
#include "allocator.hpp"
#include <vector>

namespace impl {

/** `TypeAllocatorDispatch` is useful for `DArray` to enforce the copy semantics mentioned there. */
template<typename T>
struct TypeAllocatorDispatch {
  template<typename Allocator>
  CUDA static void build(T* placement, const T& from, Allocator& allocator) {
    if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) {
      *placement = from->clone(allocator);
    }
    else if constexpr(std::is_constructible<T, const T&, const Allocator&>{}) {
      new(placement) T(from, allocator);
    }
    else {
      new(placement) T(from);
    }
  }

  template<typename Allocator>
  CUDA static void destroy(T& data, Allocator& allocator) {
    if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) {
      typedef std::remove_pointer_t<T> U;
      data->~U();
      delete data;
    }
    if constexpr(std::is_destructible_v<T>) {
      data.~T();
    }
  }
};

}

/** `DArray<T, Allocator>` is a dynamic array (so the size can depends on runtime value), but once created, the size cannot be modified (no `push_back` method).
It has some dedicated support for collection of _polymorphic objects_.
`DArray` is also usable on CPU using the `StandardAllocator` (see allocator.hpp).

In order to make the usage of collection as uniform as possible in host and device code, and with polymorphic objects or not, a `DArray` is initialized and copied following a set of rules:
- Case of non-polymorphic types `T` (i.e., `is_polymorphic_v<remove_pointer_t<T>>` is false):
  - Elements are copied using `T(const T&, const Allocator&)` if such a copy constructor is available, otherwise the normal copy constructor `T(const T&)` is used.
- Case of polymorphic type `T`: The objects of type `T` are always copied using a method `T* clone(const Allocator&)` where the `clone` method is supposed to be overloaded for the relevant/necessary allocator types.
  Note that for `T* clone(GlobalAllocator)`, we suppose the object vtable is initialized on the device.
  The polymorphic objects are automatically freed in the destructor of DArray.

Hence, _polymorphic objects are owned by DArray_.
If you use this class to store an array of polymorphic objects, the polymorphic objects will be deleted when the destructor of the array is called.
See also `tests/polymorphic_darray_test.cpp` for example on how to implement the `clone` methods and using DArray as a polymorphic container. */
template<typename T, typename Allocator>
class DArray {
  T* data_;
  size_t n;
  Allocator allocator;
public:
  typedef DArray<T, Allocator> this_type;

  /** Allocate an array of size `n` using `allocator`. */
  CUDA DArray(size_t n, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(n > 0 ? new(allocator) T[n] : nullptr) {}

  /** Default constructor. Since the size is 0 and the array cannot be extended, the allocator does not matter.*/
  CUDA DArray(): DArray(0) {}

  /** Allocate an array of size `n` using `allocator`.
      Initialize the elements of the array with those of `from`.
      NOTE: If the constructor is called from host side, with `GlobalAllocator`, then we still initialize the array in managed memory (but the polymorphic object are initialized in global memory). */
  CUDA DArray(size_t n, const T* from, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(nullptr)
  {
    if(n > 0) {
      #if ON_CPU() && __NVCC__
        if constexpr(std::is_same_v<Allocator, GlobalAllocator>) {
          data_ = new(managed_allocator) T[n];
        }
        else {
          data_ = new(allocator) T[n];
        }
      #else
        data_ = new(allocator) T[n];
      #endif
      for(size_t i = 0; i < n; ++i) {
        impl::TypeAllocatorDispatch<T>::build(&data_[i], from[i], allocator);
      }
    }
  }

  /** Copy constructor with an allocator. */
  template <typename Allocator2>
  CUDA DArray(const DArray<T, Allocator2>& from, const Allocator& alloc = Allocator()):
    DArray(from.size(), from.data(), alloc) {}

  /** Redefine the copy constructor to be sure it calls a constructor with an allocator. */
  CUDA DArray(const DArray<T, Allocator>& from): DArray(from, allocator) {}

  /** Initialize of an array of size `n` with each element initialized to `from` using `allocator`. */
  CUDA DArray(size_t n, const T& from, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(n > 0 ? new(allocator) T[n] : nullptr)
  {
    for(size_t i = 0; i < n; ++i) {
      impl::TypeAllocatorDispatch<T>::build(&data_[i], from, allocator);
    }
  }

  HOST DArray(const std::vector<T>& from, const Allocator& alloc = Allocator()):
    DArray(from.size(), from.data(), alloc) {}

  CUDA ~DArray() {
    if(data_ != nullptr) {
      for(int i = 0; i < n; ++i) {
        impl::TypeAllocatorDispatch<T>::destroy(data_[i], allocator);
      }
      operator delete[](data_, allocator);
    }
  }

  CUDA size_t size() const { return n; }
  CUDA T& operator[](size_t i) { assert(i < n); return data_[i]; }
  CUDA const T& operator[](size_t i) const { assert(i < n); return data_[i]; }
  CUDA T* data() { return data_; }
  CUDA const T* data() const { return data_; }
};

#endif
