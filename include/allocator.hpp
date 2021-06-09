// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

/** We provide a bunch of allocator compatible with the structure of the _lattice land_ project.
The allocators are aimed to be used to distinguish in which memory (shared, global, managed or the "standard" C++ memory) we should allocate data.
Use std::allocator as a standard allocator on the host side.
This allows us to provide uniform interfaces for both host (C++) and device (CUDA) code.
*/

#include <cassert>
#include <type_traits>
#include "utility.hpp"

#ifdef __NVCC__

/** A stateless allocator using the managed memory of CUDA.
This can only be used from the host side since managed memory cannot be allocated in device functions. */
class ManagedAllocator {
public:
  void* allocate(size_t bytes);
  void deallocate(void* data);
};

extern ManagedAllocator managed_allocator;

void* operator new(size_t bytes, ManagedAllocator& p);
void* operator new[](size_t bytes, ManagedAllocator& p);
void operator delete(void* ptr, ManagedAllocator& p);
void operator delete[](void* ptr, ManagedAllocator& p);

/** A stateless allocator using the global memory of CUDA.
This can be used from both the host and device side, but the memory can only be accessed when in a device function. */
class GlobalAllocator {
public:
  CUDA void* allocate(size_t bytes);
  CUDA void deallocate(void* data);
};

extern GlobalAllocator global_allocator;

CUDA void* operator new(size_t bytes, GlobalAllocator& p);
CUDA void* operator new[](size_t bytes, GlobalAllocator& p);
CUDA void operator delete(void* ptr, GlobalAllocator& p);
CUDA void operator delete[](void* ptr, GlobalAllocator& p);

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
};

CUDA void* operator new(size_t bytes, PoolAllocator& p);
CUDA void* operator new[](size_t bytes, PoolAllocator& p);
CUDA void operator delete(void* ptr, PoolAllocator& p);
CUDA void operator delete[](void* ptr, PoolAllocator& p);

#endif // ALLOCATOR_HPP
