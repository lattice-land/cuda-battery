// Copyright 2026 Ludovic Temgoua Abanda
// HIP version of simple.cpp.
// Compiled as HIP language by CMake when HIP=ON.

#include <vector>
#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/allocator.hpp"

// TODO: Remove this alias block once AMD verification is complete.
// On AMD hardware BATTERY_HIP_BACKEND is always active; replace GPU_SYNC()
// with HIPEX(hipDeviceSynchronize()) directly.
#ifdef BATTERY_CUDA_BACKEND
  #define GPU_SYNC() CUDAEX(cudaDeviceSynchronize())
#elif defined(BATTERY_HIP_BACKEND)
  #define GPU_SYNC() HIPEX(hipDeviceSynchronize())
#endif

using mvector = battery::vector<int, battery::managed_allocator>;

__global__ void kernel(mvector* v_ptr) {
  mvector& v = *v_ptr;
  // ... Compute on `v` in parallel.
}

int main(int argc, char** argv) {
  std::vector<int> v(10000, 42);
  // Transfer from CPU vector to GPU vector.
  auto gpu_v = battery::make_unique<mvector, battery::managed_allocator>(v);
  kernel<<<256, 256>>>(gpu_v.get());
  GPU_SYNC();
  // Transferring the new data to the initial vector.
  for(int i = 0; i < (int)v.size(); ++i) {
    v[i] = (*gpu_v)[i];
  }
  return 0;
}
