// Copyright 2021 Pierre Talbot, Frédéric Pinel

#include "allocator.hpp"
#include <cstdlib>


#ifdef __NVCC__

namespace battery {

void* ManagedAllocator::allocate(size_t bytes) {
  if(bytes == 0) {
    return nullptr;
  }
  void* data;
  cudaMallocManaged(&data, bytes);
  return data;
}

void ManagedAllocator::deallocate(void* data) {
  cudaFree(data);
}

ManagedAllocator managed_allocator;
GlobalAllocatorGPU global_allocator_gpu;
GlobalAllocatorCPU global_allocator_cpu;

} // namespace battery

void* operator new(size_t bytes, battery::ManagedAllocator& p) {
  return p.allocate(bytes);
}

void operator delete(void* ptr, battery::ManagedAllocator& p) {
  return p.deallocate(ptr);
}

#endif // __NVCC__

namespace battery {

CUDA PoolAllocator::PoolAllocator(const PoolAllocator& other):
  mem(other.mem), capacity(other.capacity), offset(other.offset) {}

CUDA void* PoolAllocator::allocate(size_t bytes) {
  if(bytes == 0) {
    return nullptr;
  }
  assert(offset < capacity);
  void* m = (void*)&mem[offset];
  offset += bytes / sizeof(int);
  offset += offset % sizeof(int*);
  return m;
}

CUDA void PoolAllocator::deallocate(void*) {}

void* StandardAllocator::allocate(size_t bytes) {
  return bytes == 0 ? nullptr : std::malloc(bytes);
}

void StandardAllocator::deallocate(void* data) {
  std::free(data);
}

StandardAllocator standard_allocator;

} // namespace battery


CUDA void* operator new(size_t bytes, battery::PoolAllocator& p) {
  return p.allocate(bytes);
}

CUDA void operator delete(void* ptr, battery::PoolAllocator& p) {
  return p.deallocate(ptr);
}

void* operator new(size_t bytes, battery::StandardAllocator& p) {
  return p.allocate(bytes);
}

void operator delete(void* ptr, battery::StandardAllocator& p) {
  return p.deallocate(ptr);
}
