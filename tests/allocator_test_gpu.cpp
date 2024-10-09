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
// The problem is that data is passed by value and not by reference, so
// the host constructs a temporary for shared_ptr<T> in managed memory and
// increments the reference count to 1.  Upon return, the host invokes the
// corresponding the host-side destructor for cleanup.  The problem is that
// the host's destructor tries to decrement the reference count in managed
// memory before the call to cudaDeviceSynchronize(), which is forbidden
// on a GPU that does not support concurrent managed memory access.  The
// result is a segfault on the host.
//
__global__ void kernel_managed_memory(iptr<managed_allocator> data) {
  *data = 1;
} // If the GPU does not support concurrent access this will segfault due to decrementing the ref count on the host prematurely before cudaDeviceSynchronize().

void managed_memory_test() {
  iptr<managed_allocator> data = make_shared<int, managed_allocator>(0);
  kernel_managed_memory<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
  std::cout << *data << std::endl;
  assert(*data == 1);
}

__global__ void kernel_allocate_global_memory(shared_ptr<iptr<global_allocator>, managed_allocator> data) {
  *data = make_shared<int, global_allocator>(1);
}

/** This kernel deallocate the memory contained by this `shared_ptr`. */
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
  shared_ptr<iptr<global_allocator>, managed_allocator> data = make_shared<iptr<global_allocator>, managed_allocator>(nullptr);
  kernel_allocate_global_memory<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
  kernel_test_global_memory<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
}

void global_memory_test_passing2() {
  shared_ptr<iptr<global_allocator>, managed_allocator> data = make_shared<iptr<global_allocator>, managed_allocator>(nullptr);
  kernel_allocate_global_memory<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
  kernel_test_global_memory2<<<1, 1>>>(*data);
  CUDAEX(cudaDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
}

__global__ void kernel_allocate_global_vector(shared_ptr<vector<int, global_allocator>, managed_allocator> data) {
  *data = vector<int, global_allocator>(10);
  (*data)[5] = 10;
}

__global__ void kernel_test_global_vector(vector<int, global_allocator>& data) {
  assert(data[5] == 10);
}

void global_memory_vector_passing() {
  shared_ptr<vector<int, global_allocator>, managed_allocator> data =
    make_shared<vector<int, global_allocator>, managed_allocator>(vector<int, global_allocator>{});
  kernel_allocate_global_vector<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
  kernel_test_global_vector<<<1, 1>>>(*data);
  CUDAEX(cudaDeviceSynchronize());
  kernel_free_global_memory<<<1, 1>>>(data);
  CUDAEX(cudaDeviceSynchronize());
}

// Use size * 4 + 8 bytes of memory + some possible alignment overhead for the two integers of the shared_ptr allocated independently.
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
  // Allocate 1MB in global memory.
  void* global_mem = global_allocator{}.allocate(1000000);
  pool_allocator pool_mem(static_cast<unsigned char *>(global_mem), 1000000, alignof(int));
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
  shared_ptr<int, managed_allocator> measured_mem_usage = make_shared<int, managed_allocator>(0);
  kernel_measure_memory<<<1, 1>>>(*measured_mem_usage, mem_usage);
  CUDAEX(cudaDeviceSynchronize());
  printf("measured mem usage: %d bytes\n", *measured_mem_usage);
  kernel_compute<<<1, 1, *measured_mem_usage>>>(*measured_mem_usage, mem_usage);
  CUDAEX(cudaDeviceSynchronize());
}

void shared_memory_with_precomputation() {
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel_compute);
  printf("max dynamic shared memory size: %d bytes\n", attr.maxDynamicSharedSizeBytes);
  shared_memory_max_usage(attr.maxDynamicSharedSizeBytes);
}

int main() {
  int dev = 0;
  int supportsConcurrentManagedAccess = 0;
  CUDAEX(cudaDeviceGetAttribute(&supportsConcurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, dev));
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
  CUDAEX(cudaDeviceSynchronize());
  return 0;
}
