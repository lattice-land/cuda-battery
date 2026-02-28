// HIP version of unique_ptr_test_gpu.cpp.
// Compiled as HIP language by CMake (files matching *_hip.cpp in HIP=ON builds).
//
// On AMD hardware (hip-debug):            hipcc → BATTERY_HIP_BACKEND active.
// On NVIDIA hardware (hip-nvidia-debug):  CMake's HIP language uses nvcc →
//   BATTERY_CUDA_BACKEND active, CUDA runtime API used transparently.
//
// The GPU_* aliases below pick the correct backend API at compile time.

#include <iostream>
#include <cassert>
#include "battery/allocator.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/utility.hpp"
#include "battery/vector.hpp"

// TODO: Remove this alias block once AMD verification is complete.
// On AMD hardware BATTERY_HIP_BACKEND is always active, so the #ifdef dispatch
// is dead code. Replace every GPU_* call site with the direct HIP call and
// inline gpu_launch_cooperative (note: hip/cudaLaunchCooperativeKernel differ
// in signature so it cannot be removed entirely, just de-abstracted).
//----- Backend-transparent aliases ------------------------------------------
#ifdef BATTERY_CUDA_BACKEND
  #define GPU_SYNC() CUDAEX(cudaDeviceSynchronize())
  #define GPU_DEV_ATTR(out, attr, dev) CUDAEX(cudaDeviceGetAttribute(&(out), attr, dev))
  #define GPU_ATTR_COOP_LAUNCH  cudaDevAttrCooperativeLaunch

  template<class Fn>
  void gpu_launch_cooperative(Fn* fn, dim3 grid, dim3 block, void** args) {
    CUDAEX(cudaLaunchCooperativeKernel((void*)fn, grid, block, args));
  }
#elif defined(BATTERY_HIP_BACKEND)
  #define GPU_SYNC() HIPEX(hipDeviceSynchronize())
  #define GPU_DEV_ATTR(out, attr, dev) HIPEX(hipDeviceGetAttribute(&(out), attr, dev))
  #define GPU_ATTR_COOP_LAUNCH  hipDevAttrCooperativeLaunch

  template<class Fn>
  void gpu_launch_cooperative(Fn* fn, dim3 grid, dim3 block, void** args) {
    HIPEX(hipLaunchCooperativeKernel((void*)fn, grid, block, args, 0, nullptr));
  }
#endif
// -------------------------------------------------------------------------------

using namespace battery;

__global__ void block_kernel(vector<vector<int, managed_allocator>, managed_allocator>* v_ptr)
{
  unique_ptr<int, global_allocator> block_data;
  int& data = make_unique_block(block_data, 100);
  (*v_ptr)[blockIdx.x][threadIdx.x] = data;
  __syncthreads();
  if(threadIdx.x == 0) {
    data = 1000;
  }
  __syncthreads();
  (*v_ptr)[blockIdx.x][threadIdx.x] += data;
  // This last `__syncthreads()` is needed to avoid destructing `unique_ptr` before all threads finished using `data`.
  __syncthreads();
}

void make_unique_block_test() {
  auto vptr = make_unique<vector<vector<int, managed_allocator>, managed_allocator>, managed_allocator>(10, vector<int, managed_allocator>(10));
  block_kernel<<<10, 10>>>(vptr.get());
  GPU_SYNC();
  for(int i = 0; i < 10; ++i) {
    for(int j = 0; j < 10; ++j) {
      if((*vptr)[i][j] != 1100) {
        printf("(*vptr)[%d][%d] (= %d) != 1100 \n", i, j, (*vptr)[i][j]);
        assert(false);
      }
    }
  }
}

__global__ void grid_kernel(vector<int, managed_allocator>* v_ptr) {
  auto grid = cooperative_groups::this_grid();
  unique_ptr<int, global_allocator> grid_data;
  int& data = make_unique_grid(grid_data, 100);
  (*v_ptr)[blockIdx.x] = data;
  grid.sync();
  if(blockIdx.x == 0) {
    data = 1000;
  }
  grid.sync();
  (*v_ptr)[blockIdx.x] += data;
  grid.sync();
}

void make_unique_grid_test() {
  auto vptr = make_unique<vector<int, managed_allocator>, managed_allocator>(10);
  auto ptr = vptr.get();
  void *kernelArgs[] = { &ptr };
  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(10, 1, 1);
  gpu_launch_cooperative(grid_kernel, dimGrid, dimBlock, kernelArgs);
  GPU_SYNC();
  for(int i = 0; i < 10; ++i) {
    if((*vptr)[i] != 1100) {
      printf("(*vptr)[%d] (= %d) != 1100 \n", i, (*vptr)[i]);
      assert(false);
    }
  }
}

int main() {
  make_unique_block_test();
  int supportsCoopLaunch = 0;
  GPU_DEV_ATTR(supportsCoopLaunch, GPU_ATTR_COOP_LAUNCH, 0);
  if(supportsCoopLaunch == 1) {
    make_unique_grid_test();
  }
  else {
    printf("Note: skipping unique_ptr grid test because device does not support cooperative launch.\n");
  }
  printf("unique_ptr_test_gpu_hip complete.\n");
  return 0;
}
