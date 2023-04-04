// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

/** \file allocator.hpp
We provide several allocators compatible with the structure of the _lattice land_ project.
The allocators are aimed to be used to distinguish in which memory (shared, global, managed or the "standard" C++ memory) we should allocate data.
This allows us to provide uniform interfaces for both host (C++) and device (CUDA) code.

As a general comment, be careful to always deallocate the memory from the side you allocated it, e.g., do not allocate on the host then try to deallocate it on the device.
To avoid these kind of mistakes, you should use `battery::shared_ptr` when passing data to a CUDA kernel.
*/

#include <cassert>
#include <cstddef>
#include <type_traits>
#include "utility.hpp"

#ifdef __NVCC__

namespace battery {

/** An allocator using the global memory of CUDA.
This can be used from both the host and device side, but the memory can only be accessed when in a device function.
Beware that allocation and deallocation must occur on the same side. */
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
      return nullptr;
    }
    return data;
  }

  CUDA void deallocate(void* data) {
    cudaFree(data);
  }
};

/** An allocator using the managed memory of CUDA when the memory is allocated on the host.
 * We delegate the allocation to `GlobalAllocator` when the allocation is done on the device (since managed memory cannot be allocated in device functions). */
class ManagedAllocator {
public:
  CUDA void* allocate(size_t bytes) {
    #ifdef __CUDA_ARCH__
      return GlobalAllocator{}.allocate(bytes);
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
      return GlobalAllocator{}.deallocate(data);
    #else
      cudaFree(data);
    #endif
  }
};

} // namespace battery

CUDA inline void* operator new(size_t bytes, battery::ManagedAllocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::ManagedAllocator& p) {
  return p.deallocate(ptr);
}

CUDA inline void* operator new(size_t bytes, battery::GlobalAllocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::GlobalAllocator& p) {
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
     : mem(mem), capacity(capacity), offset(0), alignment(alignment), counter(1)
    {}

    CUDA void* allocate(size_t bytes) {
      // printf("%p: allocate %lu bytes / %lu offset / %lu capacity / %lu alignment / %p current mem\n", mem, bytes, offset, capacity, alignment, &mem[offset]);
      if(bytes == 0) {
        return nullptr;
      }
      if(size_t(&mem[offset]) % alignment != 0) { // If we are currently unaligned.
        offset += alignment - (size_t(&mem[offset]) % alignment);
      }
      assert(offset + bytes <= capacity);
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

  CUDA size_t capacity() const {
    return block->capacity;
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
    if(bytes == 0) {
      return nullptr;
    }
    #ifdef __CUDA_ARCH__
      printf("cannot use StandardAllocator in CUDA __device__ code.\n");
      assert(0);
      return nullptr;
    #else
      return std::malloc(bytes);
    #endif
  }

  CUDA void deallocate(void* data) {
    #ifdef __CUDA_ARCH__
      if(data == nullptr) {
        return;
      }
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
