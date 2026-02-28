// Copyright 2026 Ludovic Temgoua Abanda
// HIP version of inkernel_allocation.cpp.
// Compiled as HIP language by CMake when HIP=ON.

#include <vector>
#include <iostream>
#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/allocator.hpp"

// TODO: Remove this alias block once AMD verification is complete.
// On AMD hardware BATTERY_HIP_BACKEND is always active; inline the HIP calls directly.
#ifdef BATTERY_CUDA_BACKEND
  #define GPU_SYNC() CUDAEX(cudaDeviceSynchronize())
  #define GPU_DEV_ATTR(out, attr, dev) CUDAEX(cudaDeviceGetAttribute(&(out), attr, dev))
  #define GPU_ATTR_COOP_LAUNCH  cudaDevAttrCooperativeLaunch
  #define GPU_DEV_GET_LIMIT(val, lim) CUDAE(cudaDeviceGetLimit(val, lim))
  #define GPU_DEV_SET_LIMIT(lim, val) CUDAE(cudaDeviceSetLimit(lim, val))
  #define GPU_LIMIT_HEAP  cudaLimitMallocHeapSize

  template<class Fn>
  void gpu_launch_cooperative(Fn* fn, dim3 grid, dim3 block, void** args) {
    CUDAEX(cudaLaunchCooperativeKernel((void*)fn, grid, block, args));
  }
#elif defined(BATTERY_HIP_BACKEND)
  #define GPU_SYNC() HIPEX(hipDeviceSynchronize())
  #define GPU_DEV_ATTR(out, attr, dev) HIPEX(hipDeviceGetAttribute(&(out), attr, dev))
  #define GPU_ATTR_COOP_LAUNCH  hipDevAttrCooperativeLaunch
  #define GPU_DEV_GET_LIMIT(val, lim) HIPE(hipDeviceGetLimit(val, lim))
  #define GPU_DEV_SET_LIMIT(lim, val) HIPE(hipDeviceSetLimit(lim, val))
  #define GPU_LIMIT_HEAP  hipLimitMallocHeapSize

  template<class Fn>
  void gpu_launch_cooperative(Fn* fn, dim3 grid, dim3 block, void** args) {
    HIPEX(hipLaunchCooperativeKernel((void*)fn, grid, block, args, 0, nullptr));
  }
#endif

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
  GPU_DEV_GET_LIMIT(&max_heap_size, GPU_LIMIT_HEAP);
  GPU_DEV_SET_LIMIT(GPU_LIMIT_HEAP, max_heap_size * 10);
  GPU_DEV_GET_LIMIT(&max_heap_size, GPU_LIMIT_HEAP);
  printf("%%GPU_max_heap_size=%zu (%zuMB)\n", max_heap_size, max_heap_size / 1000 / 1000);
}

int main() {
  increase_heap_size();
  auto vptr = battery::make_unique<mvector, battery::managed_allocator>(100000, 42);
  auto ptr = vptr.get();

  block_vector_copy<<<256, 256>>>(ptr);
  GPU_SYNC();

  int dev = 0;
  int supportsCoopLaunch = 0;
  GPU_DEV_ATTR(supportsCoopLaunch, GPU_ATTR_COOP_LAUNCH, dev);
  if(supportsCoopLaunch) {
    void *kernelArgs[] = { &ptr };
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(256, 1, 1);
    gpu_launch_cooperative(grid_vector_copy, dimGrid, dimBlock, kernelArgs);
    GPU_SYNC();
  } else {
    std::cout << "Warning: The GPU device does not support launching a cooperative kernel." << std::endl;
  }

  mvector expected(100000, 42);
  if(expected != *ptr) {
    std::cout << "Error: the vector was modified by the kernel." << std::endl;
    return 1;
  }

  return 0;
}
