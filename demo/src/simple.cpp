// Copyright 2023 Pierre Talbot

#include <vector>
#include <string>
#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/allocator.hpp"

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
  CUDAEX(cudaDeviceSynchronize());
  // Transfering the new data to the initial vector.
  for(int i = 0; i < v.size(); ++i) {
    v[i] = (*gpu_v)[i];
  }
  return 0;
}
