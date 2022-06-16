// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

/** \file allocator.hpp
We provide a bunch of allocator compatible with the structure of the _lattice land_ project.
The allocators are aimed to be used to distinguish in which memory (shared, global, managed or the "standard" C++ memory) we should allocate data.
This allows us to provide uniform interfaces for both host (C++) and device (CUDA) code.
*/

#include <cassert>
#include <cstddef>
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
It allocates a control block using the "normal" `operator new`, where the address to the pool, its capacity and current offset are stored.
*/
class PoolAllocator {

  struct ControlBlock {
    unsigned char* mem;
    size_t capacity;
    size_t offset;
    size_t alignment;
    size_t counter;

    CUDA ControlBlock(unsigned char* mem, size_t capacity, size_t alignment)
     : mem(mem), capacity(capacity), offset(0), alignment(alignment), counter(1) {}

    CUDA void* allocate(size_t bytes) {
      if(bytes == 0) {
        return nullptr;
      }
      offset += (alignment - (((size_t)&mem[offset]) % alignment)) % alignment;
      assert(offset + bytes < capacity);
      assert((size_t)&mem[offset] % alignment == 0);
      void* m = (void*)&mem[offset];
      offset += bytes;
      return m;
    }
  };

  ControlBlock* block;

public:
  CUDA PoolAllocator(const PoolAllocator& other):
    block(other.block)
  {
    block->counter++;
  }

  CUDA PoolAllocator(unsigned char* mem, size_t capacity, size_t alignment = alignof(std::max_align_t))
   : block(::new ControlBlock(mem, capacity, alignment))
  {}

  CUDA PoolAllocator() = delete;

  CUDA ~PoolAllocator() {
    block->counter--;
    if(block->counter == 0) {
      ::delete block;
    }
  }

  CUDA size_t align_at(size_t alignment) {
    size_t old = block->alignment;
    block->alignment = alignment;
    return old;
  }

  CUDA void* allocate(size_t bytes) {
    return block->allocate(bytes);
  }

  CUDA void deallocate(void*) {}

  CUDA void print() const {
    printf("Used %lu / %lu\n", block->offset, block->capacity);
  }

  CUDA size_t used() const {
    return block->offset;
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

#endif // ALLOCATOR_HPP
