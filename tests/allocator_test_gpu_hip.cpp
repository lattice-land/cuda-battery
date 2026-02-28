// HIP version of allocator_test_gpu.cpp.
// Compiled as HIP language by CMake (files matching *_hip.cpp in HIP=ON builds).
//
// On AMD hardware (hip-debug):     hipcc is the compiler → BATTERY_HIP_BACKEND active.
// On NVIDIA hardware (hip-nvidia-debug): CMake's HIP language uses nvcc →
//   BATTERY_CUDA_BACKEND active, CUDA runtime API is used transparently.
//
// The GPU_* aliases below pick the right API at compile time so the test body
// is written once.

#include <iostream>
#include <cassert>
#include "battery/bitset.hpp"
#include "battery/allocator.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/utility.hpp"
#include "battery/vector.hpp"

// TODO: Remove this alias block once AMD verification is complete.
// On AMD hardware BATTERY_HIP_BACKEND is always active, so the #ifdef dispatch
// is dead code. Replace every GPU_* call site with the direct HIP call.
// --- Backend-transparent aliases ------------------------------------------------
#ifdef BATTERY_CUDA_BACKEND
  #define GPU_SYNC() CUDAEX(cudaDeviceSynchronize())
  #define GPU_DEV_ATTR(out, attr, dev) CUDAEX(cudaDeviceGetAttribute(&(out), attr, dev))
  #define GPU_CONC_MANAGED_ATTR  cudaDevAttrConcurrentManagedAccess
  using GpuFuncAttributes = cudaFuncAttributes;
  template<class Fn>
  inline void gpu_func_get_attributes(GpuFuncAttributes& a, Fn* f) {
    CUDAEX(cudaFuncGetAttributes(&a, f));
  }
#elif defined(BATTERY_HIP_BACKEND)
  #define GPU_SYNC() HIPEX(hipDeviceSynchronize())
  #define GPU_DEV_ATTR(out, attr, dev) HIPEX(hipDeviceGetAttribute(&(out), attr, dev))
  #define GPU_CONC_MANAGED_ATTR  hipDevAttrConcurrentManagedAccess
  using GpuFuncAttributes = hipFuncAttributes;
  template<class Fn>
  inline void gpu_func_get_attributes(GpuFuncAttributes& a, Fn* f) {
    HIPEX(hipFuncGetAttributes(&a, f));
  }
#endif
// ----------------------------------------------------------------------------------

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
  GPU_SYNC();
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
  GPU_SYNC();
  kernel_test_global_memory<<<1, 1>>>(data);
  GPU_SYNC();
  kernel_free_global_memory<<<1, 1>>>(data);
  GPU_SYNC();
}

void global_memory_test_passing2() {
  auto data = make_shared<iptr<global_allocator>, managed_allocator>(nullptr);
  kernel_allocate_global_memory<<<1, 1>>>(data);
  GPU_SYNC();
  kernel_test_global_memory2<<<1, 1>>>(*data);
  GPU_SYNC();
  kernel_free_global_memory<<<1, 1>>>(data);
  GPU_SYNC();
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
  GPU_SYNC();
  kernel_test_global_vector<<<1, 1>>>(*data);
  GPU_SYNC();
  kernel_free_global_memory<<<1, 1>>>(data);
  GPU_SYNC();
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
  GPU_SYNC();
  printf("measured mem usage: %d bytes\n", *measured_mem_usage);
  kernel_compute<<<1, 1, *measured_mem_usage>>>(*measured_mem_usage, mem_usage);
  GPU_SYNC();
}

void shared_memory_with_precomputation() {
  GpuFuncAttributes attr{};
  gpu_func_get_attributes(attr, kernel_compute);
  printf("max dynamic shared memory size: %d bytes\n", attr.maxDynamicSharedSizeBytes);
  shared_memory_max_usage(attr.maxDynamicSharedSizeBytes);
}

int main() {
  int dev = 0;
  int supportsConcurrentManagedAccess = 0;
  GPU_DEV_ATTR(supportsConcurrentManagedAccess, GPU_CONC_MANAGED_ATTR, dev);
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
  GPU_SYNC();
  return 0;
}
