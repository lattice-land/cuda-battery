// Copyright 2021 Pierre Talbot, Frédéric Pinel

#include "allocator.hpp"
#include <cstdlib>


#ifdef __NVCC__

namespace battery {

CUDA void* ManagedAllocator::allocate(size_t bytes) {
  #ifdef __CUDA_ARCH__
    printf("cannot use ManagedAllocator in CUDA __device__ code.");
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

CUDA void ManagedAllocator::deallocate(void* data) {
  #ifdef __CUDA_ARCH__
    printf("cannot use ManagedAllocator in CUDA __device__ code.");
    assert(0);
  #else
    cudaFree(data);
  #endif
}

ManagedAllocator managed_allocator;
GlobalAllocatorGPU global_allocator_gpu;
GlobalAllocatorCPU global_allocator_cpu;

} // namespace battery

CUDA void* operator new(size_t bytes, battery::ManagedAllocator& p) {
  return p.allocate(bytes);
}

CUDA void operator delete(void* ptr, battery::ManagedAllocator& p) {
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

CUDA void* StandardAllocator::allocate(size_t bytes) {
  #ifdef __CUDA_ARCH__
    printf("cannot use StandardAllocator in CUDA __device__ code.");
    assert(0);
    return nullptr;
  #else
    return bytes == 0 ? nullptr : std::malloc(bytes);
  #endif
}

CUDA void StandardAllocator::deallocate(void* data) {
  #ifdef __CUDA_ARCH__
    printf("cannot use StandardAllocator in CUDA __device__ code.");
    assert(0);
  #else
    std::free(data);
  #endif
}

StandardAllocator standard_allocator;

} // namespace battery


CUDA void* operator new(size_t bytes, battery::PoolAllocator& p) {
  return p.allocate(bytes);
}

CUDA void operator delete(void* ptr, battery::PoolAllocator& p) {
  return p.deallocate(ptr);
}

CUDA void* operator new(size_t bytes, battery::StandardAllocator& p) {
  return p.allocate(bytes);
}

CUDA void operator delete(void* ptr, battery::StandardAllocator& p) {
  return p.deallocate(ptr);
}
