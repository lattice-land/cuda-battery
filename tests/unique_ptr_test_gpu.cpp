#include <iostream>
#include <cassert>
#include "battery/allocator.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/utility.hpp"
#include "battery/vector.hpp"

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
  CUDAEX(cudaDeviceSynchronize());
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
  cudaLaunchCooperativeKernel((void*)grid_kernel, dimGrid, dimBlock, kernelArgs);
  CUDAEX(cudaDeviceSynchronize());
  for(int i = 0; i < 10; ++i) {
    if((*vptr)[i] != 1100) {
      printf("(*vptr)[%d] (= %d) != 1100 \n", i, (*vptr)[i]);
      assert(false);
    }
  }
}

int main() {
  battery::configuration::gpu.init();
  make_unique_block_test();
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0);
  if(supportsCoopLaunch == 1) {
    make_unique_grid_test();
  }
  else {
    printf("Note: skipping unique_ptr grid test because device does not support cooperative launch.\n");
  }
  return 0;
}
