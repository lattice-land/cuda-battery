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
class global_allocator {
public:
  CUDA void* allocate(size_t bytes) {
    if(bytes == 0) {
      return nullptr;
    }
    #ifdef __CUDA_ARCH__
      return std::malloc(bytes);
    #else
      void* data;
      cudaError_t rc = cudaMalloc(&data, bytes);
      if (rc != cudaSuccess) {
        printf("Allocation in global memory failed (error = %d)\n", rc);
        return nullptr;
      }
      return data;
    #endif
  }

  CUDA void deallocate(void* data) {
    #ifdef __CUDA_ARCH__
      std::free(data);
    #else
      cudaFree(data);
    #endif
  }
};

/** An allocator using the managed memory of CUDA when the memory is allocated on the host.
 * We delegate the allocation to `global_allocator` when the allocation is done on the device (since managed memory cannot be allocated in device functions). */
class managed_allocator {
public:
  CUDA void* allocate(size_t bytes) {
    #ifdef __CUDA_ARCH__
      return global_allocator{}.allocate(bytes);
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
      return global_allocator{}.deallocate(data);
    #else
      cudaFree(data);
    #endif
  }
};

} // namespace battery

CUDA inline void* operator new(size_t bytes, battery::managed_allocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::managed_allocator& p) {
  return p.deallocate(ptr);
}

CUDA inline void* operator new(size_t bytes, battery::global_allocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::global_allocator& p) {
  p.deallocate(ptr);
}

#endif // __NVCC__

namespace battery {

/** An allocator allocating memory from a pool of memory.
The memory can for instance be the CUDA shared memory.
This allocator is rather incomplete as it never frees the memory allocated.
It allocates a control block using the "normal" `operator new`, where the address to the pool, its capacity and current offset are stored.
*/
class pool_allocator {

  struct control_block {
    unsigned char* mem;
    size_t capacity;
    size_t offset;
    size_t alignment;
    size_t counter;
    size_t num_deallocations;

    CUDA control_block(unsigned char* mem, size_t capacity, size_t alignment)
     : mem(mem), capacity(capacity), offset(0), alignment(alignment), num_deallocations(0), counter(1)
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

    CUDA void deallocate() {
      num_deallocations++;
    }
  };

  control_block* block;

public:
  CUDA pool_allocator(const pool_allocator& other):
    block(other.block)
  {
    block->counter++;
  }

  CUDA pool_allocator(unsigned char* mem, size_t capacity, size_t alignment = alignof(std::max_align_t))
   : block(::new control_block(mem, capacity, alignment))
  {}

  CUDA pool_allocator() = delete;

  CUDA ~pool_allocator() {
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

  CUDA void deallocate(void*) {
    block->deallocate();
  }

  CUDA void print() const {
    printf("Used %lu / %lu\n", block->offset, block->capacity);
  }

  CUDA size_t used() const {
    return block->offset;
  }

  CUDA size_t capacity() const {
    return block->capacity;
  }

  CUDA size_t num_deallocations() const {
    return block->num_deallocations;
  }
};
}

CUDA inline void* operator new(size_t bytes, battery::pool_allocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::pool_allocator& p) {
  return p.deallocate(ptr);
}

namespace battery {

/** This allocator call the standard `malloc` and `free`. */
class standard_allocator {
public:
  CUDA void* allocate(size_t bytes) {
    if(bytes == 0) {
      return nullptr;
    }
    return std::malloc(bytes);
  }

  CUDA void deallocate(void* data) {
    std::free(data);
  }
};
}

CUDA inline void* operator new(size_t bytes, battery::standard_allocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::standard_allocator& p) {
  return p.deallocate(ptr);
}

namespace battery {
  template <class Allocator, class InternalAllocator = Allocator>
  class statistics_allocator {
    struct control_block {
      Allocator allocator;
      size_t counter;
      size_t num_deallocations;
      size_t num_allocations;
      size_t total_bytes_allocated;

      CUDA control_block(const Allocator& allocator)
      : allocator(allocator), counter(1), num_deallocations(0), num_allocations(0), total_bytes_allocated(0)
      {}

      CUDA void* allocate(size_t bytes) {
        num_allocations++;
        total_bytes_allocated += bytes;
        return allocator.allocate(bytes);
      }

      CUDA void deallocate(void* ptr) {
        if(ptr != nullptr) {
          num_deallocations++;
          allocator.deallocate(ptr);
        }
      }
    };

    InternalAllocator internal_allocator;
    control_block* block;

  public:
    CUDA statistics_allocator(const statistics_allocator& other)
      : internal_allocator(other.internal_allocator), block(other.block)
    {
      block->counter++;
    }

    CUDA statistics_allocator(const Allocator& allocator = Allocator(), const InternalAllocator& internal_allocator = InternalAllocator())
      : internal_allocator(internal_allocator)
    {
      block = static_cast<control_block*>(internal_allocator.allocate(sizeof(control_block)));
      new(block) control_block(allocator);
    }

    CUDA ~statistics_allocator() {
      block->counter--;
      if(block->counter == 0) {
        internal_allocator.deallocate(block);
      }
    }

    CUDA void* allocate(size_t bytes) {
      return block->allocate(bytes);
    }

    CUDA void deallocate(void* ptr) {
      block->deallocate(ptr);
    }

    CUDA size_t num_allocations() const {
      return block->num_allocations;
    }

    CUDA size_t num_deallocations() const {
      return block->num_deallocations;
    }

    CUDA size_t total_bytes_allocated() const {
      return block->total_bytes_allocated;
    }
  };
}

#endif // ALLOCATOR_HPP
