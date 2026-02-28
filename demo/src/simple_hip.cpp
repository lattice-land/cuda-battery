// Copyright 2026 Ludovic Temgoua Abanda
// HIP version of simple.cpp.
// Compiled as HIP language by CMake when HIP=ON.

#include <vector>
#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/allocator.hpp"

// Uses the HIP API unconditionally.  On NVIDIA (hip-nvidia-*) BATTERY_HIP_BUILD is set
// by CMake so hip/hip_runtime.h is in scope; hip* calls map to cuda* via HIP headers.

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
  HIPEX(hipDeviceSynchronize());
  // Transferring the new data to the initial vector.
  for(int i = 0; i < (int)v.size(); ++i) {
    v[i] = (*gpu_v)[i];
  }
  return 0;
}
