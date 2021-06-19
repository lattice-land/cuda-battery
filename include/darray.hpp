// Copyright 2021 Pierre Talbot

#ifndef ARRAY_HPP
#define ARRAY_HPP

#include "utility.hpp"
#include "allocator.hpp"
#include <vector>

namespace impl {

#ifdef __NVCC__
  template<typename T>
  CUDA_GLOBAL void default_gpu_constructor_darray(T* array, size_t n) {
    for(size_t i = 0; i < n; ++i) {
      new(&array[i]) T;
    }
  }

  CUDA_GLOBAL void assign_at(void** array, size_t i, void* data) {
    array[i] = data;
  }

  template<typename T>
  CUDA_GLOBAL void delete_one(T** array, size_t i) {
    array[i]->~T();
    cudaFree(array[i]);
  }

  template<typename T>
  CUDA_GLOBAL void destroy(T* array, size_t i) {
    array[i].~T();
  }
#endif

#define BUILD_BODY \
  if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) { \
      T copy = from == nullptr ? nullptr : from->clone(allocator); \
      array[i] = copy; \
    } \
    else \
    { \
      if constexpr(std::is_constructible<T, const T&, const Allocator&>{}) { \
        new(&array[i]) T(from, allocator); \
      } \
      else { \
        new(&array[i]) T(from); \
      } \
    }

#define DESTROY_BODY \
  if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) { \
    typedef std::remove_pointer_t<T> U; \
    array[i]->~U(); \
    allocator.deallocate(array[i]); \
  } \
  else if constexpr(std::is_destructible_v<T>) { \
    array[i].~T(); \
  }


/** `TypeAllocatorDispatch` is useful for `DArray` to enforce the copy semantics mentioned there. */
template<typename T, typename Allocator>
struct TypeAllocatorDispatch {
  static void build(T* array, size_t i, const T& from, Allocator& allocator) {
    BUILD_BODY
  }

  static void destroy(T* array, size_t i, Allocator& allocator) {
    DESTROY_BODY
  }
};

#ifdef __NVCC__

template<typename T>
struct TypeAllocatorDispatch<T, GlobalAllocatorGPU> {
  DEVICE static void build(T* array, size_t i, const T& from, GlobalAllocatorGPU& allocator) {
    typedef GlobalAllocatorGPU Allocator;
    BUILD_BODY
  }

  DEVICE static void destroy(T* array, size_t i, GlobalAllocatorGPU& allocator) {
    DESTROY_BODY
  }
};

template<typename T>
struct TypeAllocatorDispatch<T, GlobalAllocatorCPU> {
  static void build(T* array, size_t i, const T& from, GlobalAllocatorCPU& allocator) {
    if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) {
      T copy = from == nullptr ? nullptr : from->clone(allocator);
      impl::assign_at<<<1,1>>>((void**)array, i, copy);
      CUDIE(cudaDeviceSynchronize());
      return;
    }
    else
    {
      assert(0); // "Initializing an array of non-polymorphic elements in global memory from the CPU side is not supported. Please, initialize the array directly in a kernel or use managed memory.");
      return;
    }
  }

  static void destroy(T* array, size_t i, GlobalAllocatorCPU& allocator) {
    if constexpr(std::is_pointer_v<T> && std::is_polymorphic_v<std::remove_pointer_t<T>>) {
      impl::delete_one<<<1,1>>>(array, i);
      CUDIE(cudaDeviceSynchronize());
    }
    else if constexpr(std::is_destructible_v<T>) {
      impl::destroy<<<1,1>>>(array, i);
      CUDIE(cudaDeviceSynchronize());
    }
  }
};

#endif
}

/** `DArray<T, Allocator>` is a dynamic array (so the size can depends on runtime value), but once created, the size cannot be modified (e.g., no `push_back` method).
It has some dedicated support for collection of _polymorphic objects_.
`DArray` is also usable on CPU using `StandardAllocator` (see allocator.hpp).

In order to make the usage of collection as uniform as possible in host and device code, and with polymorphic objects or not, a `DArray` is initialized and copied following this set of rules:
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

  #pragma nv_exec_check_disable
  CUDA void allocate() {
    data_ = static_cast<T*>(allocator.allocate(sizeof(T) * n));
  }

  #pragma nv_exec_check_disable
  CUDA void deallocate() {
    allocator.deallocate(data_);
  }

#ifdef __NVCC__
  bool shared = false;
  DEVICE DArray(T* data, size_t n, bool shared):
    data_(data), n(n), shared(shared), allocator(Allocator()) {}
#endif

  template<typename T2, typename Allocator2> friend class DArray;

public:
  typedef DArray<T, Allocator> this_type;

#ifdef __NVCC__
  DEVICE DArray<T, GlobalAllocatorGPU> viewOnGPU() {
    if constexpr(!(std::is_same_v<Allocator, GlobalAllocatorCPU>)) {
      assert(0);
    }
    return DArray<T, GlobalAllocatorGPU>(data_, n, true);
  }
#endif

  #pragma nv_exec_check_disable
  /** Allocate an array of size `n` using `allocator`. */
  CUDA DArray(size_t n, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(nullptr)
  {
    if(n > 0) {
      allocate();
      // If the allocator is global, and we are currently on the CPU, then we cannot initialize the elements of the array unless we are ourselves on the GPU.
      // Hence, we run a GPU kernel to initialize the vector.
      #if defined(__NVCC__)
        if constexpr(std::is_same_v<Allocator, GlobalAllocatorCPU>) {
          impl::default_gpu_constructor_darray<<<1,1>>>(data_, n);
          CUDIE(cudaDeviceSynchronize());
          return;
        }
      #endif
      T def = T();
      for(size_t i = 0; i < n; ++i) {
        impl::TypeAllocatorDispatch<T, Allocator>::build(data_, i, def, allocator);
      }
    }
  }

  /** Default constructor. Since the size is 0 and the array cannot be extended, the allocator does not matter.*/
  CUDA DArray(): DArray(0) {}

  /** Allocate an array of size `n` using `allocator`.
      Initialize the elements of the array with those of `from`. */
  #pragma nv_exec_check_disable
  CUDA DArray(size_t n, const T* from, const Allocator& alloc = Allocator()):
    n(n), allocator(alloc), data_(nullptr)
  {
    allocate();
    for(size_t i = 0; i < n; ++i) {
      impl::TypeAllocatorDispatch<T, Allocator>::build(data_, i, from[i], allocator);
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
    n(n), allocator(alloc), data_(nullptr)
  {
    allocate();
    for(size_t i = 0; i < n; ++i) {
      impl::TypeAllocatorDispatch<T, Allocator>::build(data_, i, from, allocator);
    }
  }

  HOST DArray(const std::vector<T>& from, const Allocator& alloc = Allocator()):
    DArray(from.size(), from.data(), alloc) {}

  #pragma nv_exec_check_disable
  CUDA ~DArray() {
    #ifdef __NVCC__
      if(shared) return;
    #endif
    if(data_ != nullptr) {
      assert(n > 0);
      for(int i = 0; i < n; ++i) {
        impl::TypeAllocatorDispatch<T, Allocator>::destroy(data_, i, allocator);
      }
      deallocate();
    }
    else {
      assert(n == 0);
    }
  }

  CUDA size_t size() const { return n; }
  CUDA T& operator[](size_t i) {
    assert(i < n);
    #if defined(__NVCC__)
      assert(!(std::is_same_v<Allocator, GlobalAllocatorCPU>));
        // "You cannot access the GPU global memory from the CPU."
    #endif
    return data_[i];
  }
  CUDA const T& operator[](size_t i) const {
    assert(i < n);
    #if defined(__NVCC__)
      assert(!(std::is_same_v<Allocator, GlobalAllocatorCPU>));
        // "You cannot access the GPU global memory from the CPU."
    #endif
    return data_[i];
  }
  CUDA T* data() { return data_; }
  CUDA const T* data() const { return data_; }
};

#endif
