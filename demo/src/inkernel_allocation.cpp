// Copyright 2023 Pierre Talbot

#include <vector>
#include <iostream>
#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/allocator.hpp"

/** Different aliases to `vector` with different allocators. */
using mvector = battery::vector<int, battery::managed_allocator>;
using gvector = battery::vector<int, battery::global_allocator>;

__global__ void block_vector_copy(mvector* v_ptr) {
  battery::unique_ptr<gvector, battery::global_allocator> v_block_ptr;
  gvector& v_block = battery::make_unique_block(v_block_ptr, *v_ptr);
  // Now each block has its own local copy of the vector `*v_ptr`.
  v_block[threadIdx.x] += blockIdx.x * threadIdx.x;
  // We must synchronize the threads at the end, in case the thread holding the pointer in `unique_ptr` terminates before the other.
  cooperative_groups::this_thread_block().sync(); // Alternatively, `__syncthreads();`
}

__global__ void grid_vector_copy(mvector* v_ptr) {
  battery::unique_ptr<gvector, battery::global_allocator> v_copy_ptr;
  gvector& v_copy = battery::make_unique_grid(v_copy_ptr, *v_ptr);
  // `v_copy` is now accessible by all blocks.
  v_copy[blockIdx.x * blockDim.x + threadIdx.x] += blockIdx.x * threadIdx.x;
  // Same as with block-local memory, we want to guard against destructing the pointer too early.
  cooperative_groups::this_grid().sync();
}


void increase_heap_size() {
  size_t max_heap_size;
  cudaDeviceGetLimit(&max_heap_size, cudaLimitMallocHeapSize);
  CUDAE(cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_heap_size*10));
  cudaDeviceGetLimit(&max_heap_size, cudaLimitMallocHeapSize);
  printf("%%GPU_max_heap_size=%zu (%zuMB)\n", max_heap_size, max_heap_size/1000/1000);
}

int main() {
  increase_heap_size();
  auto vptr = battery::make_unique<mvector, battery::managed_allocator>(100000, 42);
  auto ptr = vptr.get();

  block_vector_copy<<<256, 256>>>(ptr);
  CUDAEX(cudaDeviceSynchronize());

  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if (supportsCoopLaunch) {
      void *kernelArgs[] = { &ptr }; // be careful, we need to take the address of the parameter we wish to pass.
      dim3 dimBlock(256, 1, 1);
      dim3 dimGrid(256, 1, 1);
      cudaLaunchCooperativeKernel((void*)grid_vector_copy, dimGrid, dimBlock, kernelArgs);
      CUDAEX(cudaDeviceSynchronize());
  } else {
    std::cout << "Warning: The GPU device does not support launching a CUDA cooperative kernel." << std:endl;
  }
  mvector expected(100000, 42);
  if(expected != *ptr) {
    std::cout << "Error: the vector was modified by the kernel." << std::endl;
    return 1;
  }

  return 0;
}
