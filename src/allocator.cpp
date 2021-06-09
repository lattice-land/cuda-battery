// Copyright 2021 Pierre Talbot, Frédéric Pinel

#include "allocator.hpp"

#ifdef __NVCC__

ManagedAllocator managed_allocator;

void* ManagedAllocator::allocate(size_t bytes) {
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

void* operator new[](size_t bytes, ManagedAllocator& p) {
  return p.allocate(bytes);
}

void operator delete(void* ptr, ManagedAllocator& p) {
  p.deallocate(ptr);
}

void operator delete[](void* ptr, ManagedAllocator& p) {
  p.deallocate(ptr);
}

GlobalAllocator global_allocator;

CUDA void* GlobalAllocator::allocate(size_t bytes) {
  void* data;
  cudaError_t rc = cudaMalloc(&data, bytes);
  if (rc != cudaSuccess) {
    printf("Could not allocate the stack (error = %d)\n", rc);
    assert(0);
  }
  return data;
}

CUDA void GlobalAllocator::deallocate(void* data) {
  cudaFree(data);
}

CUDA void* operator new(size_t bytes, GlobalAllocator& p) {
  return p.allocate(bytes);
}

CUDA void* operator new[](size_t bytes, GlobalAllocator& p) {
  return p.allocate(bytes);
}

CUDA void operator delete(void* ptr, GlobalAllocator& p) {
  p.deallocate(ptr);
}

CUDA void operator delete[](void* ptr, GlobalAllocator& p) {
  p.deallocate(ptr);
}

#endif // __NVCC__

CUDA PoolAllocator::PoolAllocator(const PoolAllocator& other):
  mem(other.mem), capacity(other.capacity), offset(other.offset) {}

CUDA void* PoolAllocator::allocate(size_t bytes) {
  assert(offset < capacity);
  void* m = (void*)&mem[offset];
  offset += bytes / sizeof(int);
  offset += offset % sizeof(int*);
  return m;
}

CUDA void* operator new(size_t bytes, PoolAllocator& p) {
  return p.allocate(bytes);
}

CUDA void* operator new[](size_t bytes, PoolAllocator& p) {
  return p.allocate(bytes);
}

// For now, we don't support freeing the memory in the pool.
CUDA void operator delete(void* ptr, PoolAllocator& p) {}
CUDA void operator delete[](void* ptr, PoolAllocator& p) {}

StandardAllocator standard_allocator;

void* StandardAllocator::allocate(size_t bytes) {
 return ::operator new(bytes);
}

void StandardAllocator::deallocate(void* data) {
  ::operator delete(data);
}

void* operator new(size_t bytes, StandardAllocator& p) {
  return p.allocate(bytes);
}

void* operator new[](size_t bytes, StandardAllocator& p) {
  return p.allocate(bytes);
}

void operator delete(void* ptr, StandardAllocator& p) {
  p.deallocate(ptr);
}

void operator delete[](void* ptr, StandardAllocator& p) {
  p.deallocate(ptr);
}
