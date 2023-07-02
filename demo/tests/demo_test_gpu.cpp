// Copyright 2023 Pierre Talbot

#include "par_map.hpp"
#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/allocator.hpp"

using mvector = battery::vector<int, battery::managed_allocator>;

__global__ void map_kernel_test(mvector* v_ptr) {
  grid_par_map(*v_ptr, [](int x){ return x * 2; });
}

void test_with(size_t num_blocks, size_t num_threads) {
  auto gpu_v = battery::make_unique<mvector, battery::managed_allocator>(10000, 50);
  map_kernel_test<<<num_blocks, num_threads>>>(gpu_v.get());
  CUDAEX(cudaDeviceSynchronize());
  for(int i = 0; i < gpu_v->size(); ++i) {
    if((*gpu_v)[i] != 100) {
      printf("Error at index %d: %d != 100\n", i, (*gpu_v)[i]);
      exit(1);
    }
  }
  printf("demo_test_gpu [%zu blocks, %zu threads] succeeded\n", num_blocks, num_threads);
}

int main() {
  test_with(1, 1);
  test_with(1, 256);
  test_with(256, 1);
  test_with(50, 100);
  test_with(100, 100);
  test_with(100, 101);
  test_with(256, 256);
  return 0;
}
