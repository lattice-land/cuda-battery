// Copyright 2021 Pierre Talbot

#ifndef CUDA_BATTERY_ALLOCATOR_HPP
#define CUDA_BATTERY_ALLOCATOR_HPP

/** \file allocator.hpp
We provide several allocators compatible with the data structures provided by this library.
The allocators are aimed to be used to distinguish in which memory (shared, global, managed or the "standard" C++ memory) we should allocate data.
This allows us to provide uniform interfaces for both host (C++) and device (CUDA) code.

As a general comment, be careful to always deallocate the memory from the side you allocated it, e.g., do not allocate on the host then try to deallocate it on the device.
To avoid these kind of mistakes, you should use `battery::shared_ptr` when passing data to a CUDA kernel, see the manual for examples.
*/

#include <cassert>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <inttypes.h>
#include "utility.hpp"

namespace battery {

/** This allocator call the standard `malloc` and `free` functions. */
class standard_allocator {
public:
  CUDA NI void* allocate(size_t bytes) {
    if(bytes == 0) {
      return nullptr;
    }
    return std::malloc(bytes);
  }

  CUDA NI void deallocate(void* data) {
    std::free(data);
  }

  CUDA bool operator==(const standard_allocator&) const { return true; }
};
} // namespace battery

CUDA inline void* operator new(size_t bytes, battery::standard_allocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::standard_allocator& p) {
  return p.deallocate(ptr);
}

#ifdef __CUDACC__

namespace battery {

/** An allocator using the global memory of CUDA.
This can be used from both the host and device side, but the memory can only be accessed when in a device function.
Beware that allocation and deallocation must occur on the same side. */
class global_allocator {
public:
  CUDA NI void* allocate(size_t bytes) {
    if(bytes == 0) {
      return nullptr;
    }
    #ifdef __CUDA_ARCH__
      void* data = std::malloc(bytes);
      if (data == nullptr) {
        printf("Allocation of device memory failed\n");
      }
      return data;
    #else
      void* data = nullptr;
      cudaError_t rc = cudaMalloc(&data, bytes);
      if (rc != cudaSuccess) {
        std::cerr << "Allocation of global memory failed: " << cudaGetErrorString(rc) << std::endl;
        return nullptr;
      }
      return data;
    #endif
  }

  CUDA NI void deallocate(void* data) {
    #ifdef __CUDA_ARCH__
      std::free(data);
    #else
      cudaError_t rc = cudaFree(data);
      if (rc != cudaSuccess) {
        std::cerr << "Free of global memory failed: " << cudaGetErrorString(rc) << std::endl;
      }
    #endif
  }

  CUDA bool operator==(const global_allocator&) const { return true; }
};

/** An allocator using the managed memory of CUDA when the memory is allocated on the host. */
class managed_allocator {
public:
  CUDA NI void* allocate(size_t bytes) {
    #ifdef __CUDA_ARCH__
      printf("Managed memory cannot be allocated in device functions.\n");
      assert(false);
      return nullptr;
    #else
      if(bytes == 0) {
        return nullptr;
      }
      void* data = nullptr;
      cudaError_t rc = cudaMallocManaged(&data, bytes);
      if (rc != cudaSuccess) {
        std::cerr << "Allocation of managed memory failed: " << cudaGetErrorString(rc) << std::endl;
        return nullptr;
      }
      return data;
    #endif
  }

  CUDA NI void deallocate(void* data) {
    #ifdef __CUDA_ARCH__
      printf("Managed memory cannot be freed in device functions.\n");
      assert(false);
    #else
      cudaError_t rc = cudaFree(data);
      if (rc != cudaSuccess) {
        std::cerr << "Free of managed memory failed: " << cudaGetErrorString(rc) << std::endl;
      }
    #endif
  }

  CUDA bool operator==(const managed_allocator&) const { return true; }
};

/** An allocator using pinned memory for shared access between the host and the device.
 * This type of memory is required on Microsoft Windows, on the Windows Subsystem for Linux (WSL), and on NVIDIA GRID (virtual GPU), because these systems do not support concurrent access to managed memory while a CUDA kernel is running.
 *
 * This allocator requires that you first set cudaDeviceMapHost using  cudaSetDeviceFlags.
 *
 * We suppose unified virtual addressing (UVA) is enabled (the property `unifiedAddressing` is true).
 *
 * We delegate the allocation to `global_allocator` when the allocation is done on the device, since host memory cannot be allocated in device functions.
 * */
class pinned_allocator {
public:
  CUDA NI void* allocate(size_t bytes) {
    #ifdef __CUDA_ARCH__
      return global_allocator{}.allocate(bytes);
    #else
      if(bytes == 0) {
        return nullptr;
      }
      void* data = nullptr;
      cudaError_t rc = cudaMallocHost(&data, bytes); // pinned
      if (rc != cudaSuccess) {
        std::cerr << "Allocation of pinned memory failed: " << cudaGetErrorString(rc) << std::endl;
        return nullptr;
      }
      return data;
    #endif
  }

  CUDA NI void deallocate(void* data) {
    #ifdef __CUDA_ARCH__
      return global_allocator{}.deallocate(data);
    #else
      cudaError_t rc = cudaFreeHost(data);
      if (rc != cudaSuccess) {
        std::cerr << "Free of pinned memory failed: " << cudaGetErrorString(rc) << std::endl;
      }
    #endif
  }

  CUDA bool operator==(const pinned_allocator&) const { return true; }
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

CUDA inline void* operator new(size_t bytes, battery::pinned_allocator& p) {
  return p.allocate(bytes);
}

CUDA inline void operator delete(void* ptr, battery::pinned_allocator& p) {
  p.deallocate(ptr);
}

#endif // __CUDACC__

namespace battery {

namespace impl {
#ifdef __CUDA_ARCH__
  __constant__
#endif
static const int power2[17] = {0, 1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16};
}

/** An allocator allocating memory from a pool of memory.
The memory can for instance be the CUDA shared memory.
This allocator is incomplete as it never frees the memory allocated.
It allocates a control block using the "normal" `operator new`, where the address to the pool, its capacity and current offset are stored.
*/
class pool_allocator {

  struct control_block {
    unsigned char* mem;
    size_t capacity;
    size_t offset;
    size_t alignment;
    size_t num_deallocations;
    size_t num_allocations;
    size_t unaligned_wasted_bytes;
    size_t counter;

    CUDA control_block(unsigned char* mem, size_t capacity, size_t alignment)
     : mem(mem), capacity(capacity), offset(0), alignment(alignment), num_deallocations(0), num_allocations(0), unaligned_wasted_bytes(0), counter(1)
    {}

    CUDA void* allocate(size_t bytes) {
      // printf("%p: allocate %lu bytes / %lu offset / %lu capacity / %lu alignment / %p current mem\n", mem, bytes, offset, capacity, alignment, &mem[offset]);
      if(bytes == 0) {
        return nullptr;
      }
      size_t smallest_alignment = (bytes > alignment || alignment > 16) ? alignment : impl::power2[bytes];
      if(size_t(&mem[offset]) % smallest_alignment != 0) { // If we are currently unaligned.
        size_t old_offset = offset;
        offset += smallest_alignment - (size_t(&mem[offset]) % smallest_alignment);
        unaligned_wasted_bytes += (offset - old_offset);
      }
      assert(offset + bytes <= capacity);
      assert((size_t)&mem[offset] % smallest_alignment == 0);
      void* m = (void*)&mem[offset];
      offset += bytes;
      num_allocations++;
      return m;
    }

    CUDA void deallocate(void* ptr) {
      if(ptr != nullptr) {
        num_deallocations++;
      }
    }
  };

  control_block* block;

public:
  NI pool_allocator() = default;

  CUDA NI pool_allocator(const pool_allocator& other):
    block(other.block)
  {
    if(block != nullptr) {
      block->counter++;
    }
  }

  CUDA NI pool_allocator(pool_allocator&& other):
    block(other.block)
  {
    other.block = nullptr;
  }

  CUDA NI pool_allocator(unsigned char* mem, size_t capacity, size_t alignment = alignof(std::max_align_t))
   : block(new control_block(mem, capacity, alignment))
  {}

private:
  CUDA void destroy() {
    block->counter--;
    if(block->counter == 0) {
      // This is a temporary hack to disable deleting the block when using Turbo...
      // Unfortunately, there is a bug with -arch gpu and it seems the block is deleted from host while allocating on device (or vice-versa).
      #ifdef CUDA_THREADS_PER_BLOCK
      #else
        delete block;
      #endif
    }
  }

public:
  CUDA NI ~pool_allocator() {
    if(block != nullptr) {
      destroy();
    }
  }

  CUDA NI pool_allocator& operator=(pool_allocator&& other) {
    if(block != nullptr) {
      destroy();
    }
    block = other.block;
    other.block = nullptr;
    return *this;
  }

  CUDA size_t align_at(size_t alignment) {
    size_t old = block->alignment;
    block->alignment = alignment;
    return old;
  }

  CUDA NI void* allocate(size_t bytes) {
    return block->allocate(bytes);
  }

  CUDA NI void deallocate(void* ptr) {
    block->deallocate(ptr);
  }

  CUDA NI void print() const {
    // CUDA printf does not support "%zu" -- use PRIu64 macro (Windows / Linux)
    printf("%% %" PRIu64 " / %" PRIu64 " used [%" PRIu64 "/%" PRIu64 "]KB [%" PRIu64 "/%" PRIu64 "]MB\n",
      block->offset, block->capacity,
      block->offset/1000, block->capacity/1000,
      block->offset/1000/1000, block->capacity/1000/1000);
    printf("%% %" PRIu64 " / %" PRIu64 " wasted for alignment [%" PRIu64 "/%" PRIu64 "]KB [%" PRIu64 "/%" PRIu64 "]MB\n",
      block->unaligned_wasted_bytes, block->offset,
      block->unaligned_wasted_bytes/1000, block->offset/1000,
      block->unaligned_wasted_bytes/1000/1000, block->offset/1000/1000);
    printf("%% %" PRIu64 " allocations and %" PRIu64 " deallocations\n", block->num_allocations, block->num_deallocations);
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

  CUDA size_t num_allocations() const {
    return block->num_allocations;
  }

  CUDA size_t unaligned_wasted_bytes() const {
    return block->unaligned_wasted_bytes;
  }

  CUDA bool operator==(const pool_allocator& rhs) const {
    return block == rhs.block;
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

      CUDA NI void* allocate(size_t bytes) {
        num_allocations++;
        total_bytes_allocated += bytes;
        return allocator.allocate(bytes);
      }

      CUDA NI void deallocate(void* ptr) {
        if(ptr != nullptr) {
          num_deallocations++;
          allocator.deallocate(ptr);
        }
      }
    };

    InternalAllocator internal_allocator;
    control_block* block;

  public:
    using this_type = statistics_allocator<Allocator, InternalAllocator>;

    CUDA NI statistics_allocator(const statistics_allocator& other)
      : internal_allocator(other.internal_allocator), block(other.block)
    {
      block->counter++;
    }

    CUDA NI statistics_allocator(const Allocator& allocator = Allocator(), const InternalAllocator& internal_allocator = InternalAllocator())
      : internal_allocator(internal_allocator)
    {
      block = static_cast<control_block*>(this->internal_allocator.allocate(sizeof(control_block)));
      new(block) control_block(allocator);
    }

    CUDA NI ~statistics_allocator() {
      block->counter--;
      if(block->counter == 0) {
        internal_allocator.deallocate(block);
      }
    }

    CUDA NI void* allocate(size_t bytes) {
      return block->allocate(bytes);
    }

    CUDA NI void deallocate(void* ptr) {
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

    CUDA inline bool operator==(const this_type& rhs) const {
      return block == rhs.block;
    }
  };
}

#endif
