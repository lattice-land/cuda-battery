// Copyright 2021 Pierre Talbot, Frédéric Pinel

#include "allocator.hpp"
#include <cstdlib>

#ifdef __NVCC__

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

void* operator new(size_t bytes, ManagedAllocator& p) {
  return p.allocate(bytes);
}

void operator delete(void* ptr, ManagedAllocator& p) {
  return p.deallocate(ptr);
}

ManagedAllocator managed_allocator;
GlobalAllocatorGPU global_allocator_gpu;
GlobalAllocatorCPU global_allocator_cpu;

#endif // __NVCC__

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

CUDA void* operator new(size_t bytes, PoolAllocator& p) {
  return p.allocate(bytes);
}

CUDA void operator delete(void* ptr, PoolAllocator& p) {
  return p.deallocate(ptr);
}

void* StandardAllocator::allocate(size_t bytes) {
  return bytes == 0 ? nullptr : std::malloc(bytes);
}

void StandardAllocator::deallocate(void* data) {
  std::free(data);
}

void* operator new(size_t bytes, StandardAllocator& p) {
  return p.allocate(bytes);
}

void operator delete(void* ptr, StandardAllocator& p) {
  return p.deallocate(ptr);
}

StandardAllocator standard_allocator;
