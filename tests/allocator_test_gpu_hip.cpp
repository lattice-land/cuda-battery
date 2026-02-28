// HIP version of allocator_test_gpu.cpp.
// Compiled as LANGUAGE HIP by CMake (files matching *_hip.cpp in HIP=ON builds).
//
// Uses the HIP API unconditionally.  On NVIDIA (hip-nvidia-*) the BATTERY_HIP_BUILD
// define is set by CMake so hip/hip_runtime.h is in scope and every hip* call maps
// transparently to the corresponding cuda* call via the HIP portability headers.
// On AMD (hip-debug) these are native HIP calls.

#include <iostream>
#include <cassert>
#include "battery/bitset.hpp"
#include "battery/allocator.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/utility.hpp"
#include "battery/vector.hpp"

using namespace battery;

template <class Alloc>
using iptr = shared_ptr<int, Alloc>;

//
// Note: These tests do not work on a GPU that does not support concurrent
// access to managed memory.
//
__global__ void kernel_managed_memory(iptr<managed_allocator> data) {
  *data = 1;
}

void managed_memory_test() {
  iptr<managed_allocator> data = make_shared<int, managed_allocator>(0);
  kernel_managed_memory<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
  std::cout << *data << std::endl;
  assert(*data == 1);
}

__global__ void kernel_allocate_global_memory(shared_ptr<iptr<global_allocator>, managed_allocator> data) {
  *data = make_shared<int, global_allocator>(1);
}

template <class T>
__global__ void kernel_free_global_memory(shared_ptr<T, managed_allocator> data) {
  *data = T{};
}

__global__ void kernel_test_global_memory(shared_ptr<iptr<global_allocator>, managed_allocator> data) {
  assert(**data == 1);
}

__global__ void kernel_test_global_memory2(iptr<global_allocator>& data) {
  assert(*data == 1);
}

void global_memory_test_passing1() {
  auto data = make_shared<iptr<global_allocator>, managed_allocator>(nullptr);
  kernel_allocate_global_memory<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
  kernel_test_global_memory<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
}

void global_memory_test_passing2() {
  auto data = make_shared<iptr<global_allocator>, managed_allocator>(nullptr);
  kernel_allocate_global_memory<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
  kernel_test_global_memory2<<<1, 1>>>(*data);
  HIPEX(hipDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
}

__global__ void kernel_allocate_global_vector(shared_ptr<vector<int, global_allocator>, managed_allocator> data) {
  *data = vector<int, global_allocator>(10);
  (*data)[5] = 10;
}

__global__ void kernel_test_global_vector(vector<int, global_allocator>& data) {
  assert(data[5] == 10);
}

void global_memory_vector_passing() {
  auto data = make_shared<vector<int, global_allocator>, managed_allocator>(vector<int, global_allocator>{});
  kernel_allocate_global_vector<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
  kernel_test_global_vector<<<1, 1>>>(*data);
  HIPEX(hipDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  HIPEX(hipDeviceSynchronize());
}

CUDA void use_memory(const pool_allocator& pool_mem, int size) {
  vector<int, pool_allocator> data(size, pool_mem);
  for(int i = 0; i < size; ++i) {
    data[i] = i;
  }
  shared_ptr<int, pool_allocator> min = allocate_shared<int, pool_allocator>(pool_mem, size);
  for(int i = 0; i < size; ++i) {
    *min = battery::min(*min, data[i]);
  }
  assert(*min == 0);
}

__global__ void kernel_measure_memory(int& measured_mem_usage, int mem_usage) {
  void* global_mem = global_allocator{}.allocate(1000000);
  pool_allocator pool_mem(static_cast<unsigned char*>(global_mem), 1000000, alignof(int));
  use_memory(pool_mem, mem_usage);
  measured_mem_usage = pool_mem.used();
  global_allocator{}.deallocate(global_mem);
}

__global__ void kernel_compute(int measured_mem_usage, int mem_usage) {
  extern __shared__ unsigned char shared_mem[];
  pool_allocator pool_mem(shared_mem, measured_mem_usage, alignof(int));
  use_memory(pool_mem, mem_usage);
  assert(measured_mem_usage == pool_mem.used());
}

__global__ void test_empty_pool() {
  shared_ptr<int, pool_allocator> x(pool_allocator(nullptr, 0));
  void* mem_pool = global_allocator{}.allocate(10);
  pool_allocator pool(static_cast<unsigned char*>(mem_pool), 10);
  x = allocate_shared<int, pool_allocator>(pool);
}

void shared_memory_max_usage(int shared_memory_size) {
  const int mem_usage = shared_memory_size / 4 - 2;
  auto measured_mem_usage = make_shared<int, managed_allocator>(0);
  kernel_measure_memory<<<1, 1>>>(*measured_mem_usage, mem_usage);
  HIPEX(hipDeviceSynchronize());
  printf("measured mem usage: %d bytes\n", *measured_mem_usage);
  kernel_compute<<<1, 1, *measured_mem_usage>>>(*measured_mem_usage, mem_usage);
  HIPEX(hipDeviceSynchronize());
}

void shared_memory_with_precomputation() {
  hipFuncAttributes attr{};
  HIPEX(hipFuncGetAttributes(&attr, (const void*)kernel_compute));
  printf("max dynamic shared memory size: %d bytes\n", attr.maxDynamicSharedSizeBytes);
  shared_memory_max_usage(attr.maxDynamicSharedSizeBytes);
}

int main() {
  int dev = 0;
  int supportsConcurrentManagedAccess = 0;
  HIPEX(hipDeviceGetAttribute(&supportsConcurrentManagedAccess, hipDeviceAttributeConcurrentManagedAccess, dev));
  if (!supportsConcurrentManagedAccess) {
    std::cout << "Cannot run tests because the GPU does not support concurrent access to managed memory." << std::endl;
  } else {
    managed_memory_test();
    global_memory_test_passing1();
    global_memory_test_passing2();
    global_memory_vector_passing();
  }
  shared_memory_with_precomputation();
  test_empty_pool<<<1, 1>>>();
  HIPEX(hipDeviceSynchronize());
  return 0;
}
