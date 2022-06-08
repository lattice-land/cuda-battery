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

namespace battery {

/** An allocator using the managed memory of CUDA.
This can only be used from the host side since managed memory cannot be allocated in device functions. */
class ManagedAllocator {
public:
  CUDA void* allocate(size_t bytes) {
    #ifdef __CUDA_ARCH__
      printf("cannot use ManagedAllocator in CUDA __device__ code.\n");
      assert(0);
      return nullptr;
    #else
      if(bytes == 0) {
        return nullptr;
      }
      void* data;
      cudaMallocManaged(&data, bytes);
      return data;
    #endif
  }

  CUDA void deallocate(void* data) {
    #ifdef __CUDA_ARCH__
      printf("cannot use ManagedAllocator in CUDA __device__ code.\n");
      assert(0);
    #else
      cudaFree(data);
    #endif
  }
};

/** An allocator using the global memory of CUDA.
This can be used from both the host and device side, but the memory can only be accessed when in a device function. */
template<bool on_gpu>
class GlobalAllocator {
public:
  CUDA void* allocate(size_t bytes) {
    #ifdef __CUDA_ARCH__
      assert(on_gpu);
    #else
      assert(!on_gpu);
    #endif
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
    #ifdef __CUDA_ARCH__
      assert(on_gpu);
    #else
      assert(!on_gpu);
    #endif
    cudaFree(data);
  }
};

using GlobalAllocatorGPU = GlobalAllocator<true>;
using GlobalAllocatorCPU = GlobalAllocator<false>;
} // namespace battery


CUDA inline void* operator new(size_t bytes, battery::ManagedAllocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::ManagedAllocator& p) {
  return p.deallocate(ptr);
}

template<bool on_gpu>
CUDA void* operator new(size_t bytes, battery::GlobalAllocator<on_gpu>& p) {
  return p.allocate(bytes);
}

template<bool on_gpu>
CUDA void operator delete(void* ptr, battery::GlobalAllocator<on_gpu>& p) {
  p.deallocate(ptr);
}

#endif // __NVCC__

namespace battery {

/** An allocator allocating memory from a pool of memory.
The memory can for instance be the CUDA shared memory.
This allocator is rather incomplete as it never frees the memory allocated.
It also does not care a bit about memory alignment. */
class PoolAllocator {
  unsigned char* mem;
  size_t offset;
  size_t capacity;
public:
  CUDA PoolAllocator(const PoolAllocator& other):
    mem(other.mem), capacity(other.capacity), offset(other.offset) {}

  CUDA PoolAllocator(unsigned char* mem, size_t capacity):
    mem(mem), capacity(capacity), offset(0) {}

  CUDA PoolAllocator() = delete;

  CUDA void* allocate(size_t bytes) {
    printf("Allocate %lu bytes.\n", bytes);
    if(bytes == 0) {
      return nullptr;
    }
    assert(offset < capacity);
    void* m = (void*)&mem[offset];
    offset += bytes;
    return m;
  }

  CUDA void deallocate(void*) {}

  CUDA void print() {
    printf("Used %lu / %lu\n", offset, capacity);
  }
};
}

CUDA inline void* operator new(size_t bytes, battery::PoolAllocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::PoolAllocator& p) {
  return p.deallocate(ptr);
}

namespace battery {

/** This allocator call the standard `malloc` and `free`. */
class StandardAllocator {
public:
  CUDA void* allocate(size_t bytes) {
    #ifdef __CUDA_ARCH__
      printf("cannot use StandardAllocator in CUDA __device__ code.\n");
      assert(0);
      return nullptr;
    #else
      return bytes == 0 ? nullptr : std::malloc(bytes);
    #endif
  }

  CUDA void deallocate(void* data) {
    #ifdef __CUDA_ARCH__
      printf("cannot use StandardAllocator in CUDA __device__ code.\n");
      assert(0);
    #else
      std::free(data);
    #endif
  }
};
}

CUDA inline void* operator new(size_t bytes, battery::StandardAllocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::StandardAllocator& p) {
  return p.deallocate(ptr);
}

namespace battery {

/** `A` is an allocator for a "slow but large" memory, and `B` is an allocator for a "fast but small" memory.
 * By default, the allocator `A` is used, unless `B` is explicitly asked through `fast()`. */
template <typename A, typename B>
class TradeoffAllocator {
  A a;
  B b;
public:
  using LargeMemAllocator = A;
  using FastMemAllocator = B;

  CUDA TradeoffAllocator(const A& a, const B& b): a(a), b(b) {}
  CUDA TradeoffAllocator(const B& b): a(), b(b) {}
  CUDA void* allocate(size_t bytes) {
    return a.allocate(bytes);
  }

  CUDA void deallocate(void* data) {
    return a.deallocate(data);
  }

  CUDA B& fast() { return b; }
};

template<typename A>
struct FasterAllocator {
  using type = A;
  CUDA static type& fast(A& a) { return a; }
};

template<typename A, typename B>
struct FasterAllocator<TradeoffAllocator<A, B>> {
  using type = B;
  CUDA static type& fast(TradeoffAllocator<A, B>& a) { return a.fast(); }
};

} // namespace battery

#endif // ALLOCATOR_HPP
