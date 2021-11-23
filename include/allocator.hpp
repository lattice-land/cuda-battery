// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

/** \file allocator.hpp
We provide a bunch of allocator compatible with the structure of the _lattice land_ project.
The allocators are aimed to be used to distinguish in which memory (shared, global, managed or the "standard" C++ memory) we should allocate data.
This allows us to provide uniform interfaces for both host (C++) and device (CUDA) code.
*/

#include <cassert>
#include <type_traits>
#include "utility.hpp"

#ifdef __NVCC__

/** An allocator using the managed memory of CUDA.
This can only be used from the host side since managed memory cannot be allocated in device functions. */
class ManagedAllocator {
public:
  void* allocate(size_t bytes);
  void deallocate(void* data);
};

void* operator new(size_t bytes, ManagedAllocator& p);
void operator delete(void* ptr, ManagedAllocator& p);

extern ManagedAllocator managed_allocator;

/** An allocator using the global memory of CUDA.
This can be used from both the host and device side, but the memory can only be accessed when in a device function. */
template<bool on_gpu>
class GlobalAllocator {
public:
  CUDA void* allocate(size_t bytes) {
    if(bytes == 0) {
      return nullptr;
    }
    void* data;
    cudaError_t rc = cudaMalloc(&data, bytes);
    if (rc != cudaSuccess) {
      printf("Allocation in global memory failed (error = %d)\n", rc);
      assert(0);
    }
    return data;
  }

  CUDA void deallocate(void* data) {
    cudaFree(data);
  }
};

template<bool on_gpu>
CUDA void* operator new(size_t bytes, GlobalAllocator<on_gpu>& p) {
  return p.allocate(bytes);
}

template<bool on_gpu>
CUDA void operator delete(void* ptr, GlobalAllocator<on_gpu>& p) {
  p.deallocate(ptr);
}

using GlobalAllocatorGPU = GlobalAllocator<true>;
using GlobalAllocatorCPU = GlobalAllocator<false>;
extern GlobalAllocatorGPU global_allocator_gpu;
extern GlobalAllocatorCPU global_allocator_cpu;

#endif // __NVCC__

/** An allocator allocating memory from a pool of memory.
The memory can for instance be the CUDA shared memory.
This allocator is rather incomplete as it never frees the memory allocated. */
class PoolAllocator {
  int* mem;
  size_t offset;
  size_t capacity;
public:
  CUDA PoolAllocator(const PoolAllocator&);
  CUDA PoolAllocator(int* mem, size_t capacity);
  CUDA PoolAllocator() = delete;
  CUDA void* allocate(size_t bytes);
  CUDA void deallocate(void* ptr);
};

CUDA void* operator new(size_t bytes, PoolAllocator& p);
CUDA void operator delete(void* ptr, PoolAllocator& p);

/** This allocator call the standard `malloc` and `free`. */
class StandardAllocator {
public:
  void* allocate(size_t bytes);
  void deallocate(void* data);
};

void* operator new(size_t bytes, StandardAllocator& p);
void operator delete(void* ptr, StandardAllocator& p);
extern StandardAllocator standard_allocator;

/** `A` is an allocator for a "slow but large" memory, and `B` is an allocator for a "fast but small" memory.
 * By default, the allocator `A` is used, unless `B` is explicitly asked through `fast()`. */
template <typename A, typename B>
class TradeoffAllocator {
  A a;
  B b;
public:
  using LargeMemAllocator = A;
  using FastMemAllocator = B;

  TradeoffAllocator(const A& a, const B& b): a(a), b(b) {}
  TradeoffAllocator(const B& b): a(), b(b) {}
  void* allocate(size_t bytes) {
    return a.allocate(bytes);
  }

  void deallocate(void* data) {
    return a.deallocate(data);
  }

  B& fast() { return b; }
};

template<typename A>
struct FasterAllocator {
  using type = A;
  static type& fast(A& a) { return a; }
};

template<typename A, typename B>
struct FasterAllocator<TradeoffAllocator<A, B>> {
  using type = B;
  static type& fast(TradeoffAllocator<A, B>& a) { return a.fast(); }
};

#endif // ALLOCATOR_HPP
