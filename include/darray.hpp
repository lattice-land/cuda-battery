// Copyright 2021 Pierre Talbot

#ifndef ARRAY_HPP
#define ARRAY_HPP

#include "utility.hpp"

namespace impl {

/** `TypeAllocatorDispatch` is useful for `DArray` to enforce the copy semantics mentioned there. */
template<typename T>
struct TypeAllocatorDispatch {
  template<typename Allocator>
  CUDA static void build(T* placement, const T& from, const Allocator& allocator) {
    if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) {
      *placement = from->clone_in(allocator);
    }
    if constexpr(std::is_constructible<T, const T&, const Allocator&>{}) {
      new(placement) T(from, allocator);
    }
    else {
      new(placement) T(from);
    }
  }

  template<typename Allocator>
  CUDA static void destroy(T* ptr, const Allocator& allocator) {
    if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) {
      (*ptr)->~T();
      delete *ptr;
    }
    if constexpr(std::is_destructible_v<T>) {
      ptr->~T();
    }
  }
};

}

/** `DArray<T, Allocator>` is a dynamic array (so the size can depends on runtime value), but once created, the size cannot be modified (no `push_back` method).
Note that this class is also usable on CPU using the `StandardAllocator` (see allocator.hpp).

Initialization/Copy semantics: The elements of the array are initialized by using the following of the provided parameter:
  - If `T` is polymorphic, we use a dedicated `clone_in` method, the polymorphic objects are automatically freed in the destructor of DArray.
  - `T(const T&, const Allocator&)` if such a copy constructor is available.
  - The normal copy constructor `T(const T&)` otherwise.

Hence, _polymorphic objects are owned by DArray_.
If you use this class to store an array of polymorphic objects, the polymorphic objects will be deleted when the destructor of the array is called. */
template<typename T, typename Allocator>
class DArray {
  T* data_;
  size_t n;
  Allocator allocator;
public:
  typedef DArray<T, Allocator> this_type;

  /** Allocate an array of size `n` using `allocator`. */
  CUDA DArray(size_t n, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(new(allocator) T[n]) {}

  /** Allocate an array of size `n` using `allocator`.
      Initialize the elements of the array with those of `from`.*/
  CUDA DArray(size_t n, const T* from, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(new(allocator) T[n])
  {
    for(size_t i = 0; i < n; ++i) {
      impl::TypeAllocatorDispatch<T>::build(&data_[i], from[i], allocator);
    }
  }

  /** Copy constructor with an allocator. */
  template <typename Allocator2>
  CUDA DArray(const DArray<T, Allocator2>& from, const Allocator& alloc = Allocator()):
    DArray(from.n, from.data_, alloc) {}

  /** Redefine the copy constructor to be sure it calls a constructor with an allocator. */
  CUDA DArray(const DArray<T, Allocator>& from): DArray(from, Allocator()) {}

  /** Initialize of an array of size `n` with each element initialized to `from` using `allocator`. */
  CUDA DArray(size_t n, const T& from, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(new(allocator) T[n])
  {
    for(size_t i = 0; i < n; ++i) {
      impl::TypeAllocatorDispatch<T>::build(&data_[i], from, allocator);
    }
  }

  HOST DArray(const std::vector<T>& from, const Allocator& alloc = Allocator()):
    DArray(from.size(), from.data(), alloc) {}

  CUDA ~DArray() {
    for(int i = 0; i < n; ++i) {
      impl::TypeAllocatorDispatch<T>::destroy(&data_[i], allocator);
    }
    operator delete[](data_, allocator);
  }

  CUDA size_t size() const { return n; }
  CUDA T& operator[](size_t i) { assert(i < n); return data_[i]; }
  CUDA const T& operator[](size_t i) const { assert(i < n); return data_[i]; }
  CUDA T* data() { return data_; }
  CUDA const T* data() const { return data_; }
};

#endif
