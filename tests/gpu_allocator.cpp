#include <iostream>
#include <cassert>
#include "bitset.hpp"
#include "allocator.hpp"
#include "shared_ptr.hpp"
#include "utility.hpp"
#include "vector.hpp"

using namespace battery;

template <class Alloc>
using iptr = shared_ptr<int, Alloc>;

__global__ void kernel_managed_memory(iptr<ManagedAllocator> data) {
  *data = 1;
}

void managed_memory_test() {
  iptr<ManagedAllocator> data = make_shared<int, ManagedAllocator>(0);
  kernel_managed_memory<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
  std::cout << *data << std::endl;
  assert(*data == 1);
}

__global__ void kernel_allocate_global_memory(shared_ptr<iptr<GlobalAllocator>, ManagedAllocator> data) {
  *data = make_shared<int, GlobalAllocator>(1);
}

/** This kernel deallocate the memory contained by this `shared_ptr`. */
template <class T>
__global__ void kernel_free_global_memory(shared_ptr<T, ManagedAllocator> data) {
  *data = T{};
}

__global__ void kernel_test_global_memory(shared_ptr<iptr<GlobalAllocator>, ManagedAllocator> data) {
  assert(**data == 1);
}

__global__ void kernel_test_global_memory2(iptr<GlobalAllocator>& data) {
  assert(*data == 1);
}

void global_memory_test_passing1() {
  shared_ptr<iptr<GlobalAllocator>, ManagedAllocator> data = make_shared<iptr<GlobalAllocator>, ManagedAllocator>(nullptr);
  kernel_allocate_global_memory<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
  kernel_test_global_memory<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
}

void global_memory_test_passing2() {
  shared_ptr<iptr<GlobalAllocator>, ManagedAllocator> data = make_shared<iptr<GlobalAllocator>, ManagedAllocator>(nullptr);
  kernel_allocate_global_memory<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
  kernel_test_global_memory2<<<1, 1>>>(*data);
  CUDIE(cudaDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
}

__global__ void kernel_allocate_global_vector(shared_ptr<vector<int, GlobalAllocator>, ManagedAllocator> data) {
  *data = vector<int, GlobalAllocator>(10);
  (*data)[5] = 10;
}

__global__ void kernel_test_global_vector(vector<int, GlobalAllocator>& data) {
  assert(data[5] == 10);
}

void global_memory_vector_passing() {
  shared_ptr<vector<int, GlobalAllocator>, ManagedAllocator> data =
    make_shared<vector<int, GlobalAllocator>, ManagedAllocator>(vector<int, GlobalAllocator>{});
  kernel_allocate_global_vector<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
  kernel_test_global_vector<<<1, 1>>>(*data);
  CUDIE(cudaDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  CUDIE(cudaDeviceSynchronize());
}

// Use size * 4 + 8 bytes of memory + some possible alignment overhead for the two integers of the shared_ptr allocated independently.
CUDA void use_memory(const PoolAllocator& pool_mem, int size) {
  vector<int, PoolAllocator> data(size, pool_mem);
  for(int i = 0; i < size; ++i) {
    data[i] = i;
  }
  shared_ptr<int, PoolAllocator> min = allocate_shared<int, PoolAllocator>(pool_mem, size);
  for(int i = 0; i < size; ++i) {
    *min = battery::min(*min, data[i]);
  }
  assert(*min == 0);
}

__global__ void kernel_measure_memory(int& measured_mem_usage, int mem_usage) {
  // Allocate 1MB in global memory.
  void* global_mem = GlobalAllocator{}.allocate(1000000);
  PoolAllocator pool_mem(static_cast<unsigned char *>(global_mem), 1000000, alignof(int));
  use_memory(pool_mem, mem_usage);
  measured_mem_usage = pool_mem.used();
  GlobalAllocator{}.deallocate(global_mem);
}

__global__ void kernel_compute(int measured_mem_usage, int mem_usage) {
  extern __shared__ unsigned char shared_mem[];
  PoolAllocator pool_mem(shared_mem, measured_mem_usage, alignof(int));
  use_memory(pool_mem, mem_usage);
  assert(measured_mem_usage == pool_mem.used());
}

void shared_memory_max_usage(int shared_memory_size) {
  const int mem_usage = shared_memory_size / 4 - 2;
  shared_ptr<int, ManagedAllocator> measured_mem_usage = make_shared<int, ManagedAllocator>(0);
  kernel_measure_memory<<<1, 1>>>(*measured_mem_usage, mem_usage);
  CUDIE(cudaDeviceSynchronize());
  printf("measured mem usage: %d bytes\n", *measured_mem_usage);
  kernel_compute<<<1, 1, *measured_mem_usage>>>(*measured_mem_usage, mem_usage);
  CUDIE(cudaDeviceSynchronize());
}

void shared_memory_with_precomputation() {
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel_compute);
  printf("max dynamic shared memory size: %d bytes\n", attr.maxDynamicSharedSizeBytes);
  shared_memory_max_usage(attr.maxDynamicSharedSizeBytes);
}

int main() {
  managed_memory_test();
  global_memory_test_passing1();
  global_memory_test_passing2();
  global_memory_vector_passing();
  shared_memory_with_precomputation();
  return 0;
}
