# Standard Library for CUDA Programming

[![Build Status](https://travis-ci.com/lattice-land/cuda-battery.svg?branch=main)](https://travis-ci.com/lattice-land/cuda-battery)

This library provides data structures to ease programming in CUDA (version 12 or higher).
For a tutorial and further information, please read [this manual](https://lattice-land.github.io/1-cuda-battery.html).

## Example

Quick example on how to transfer a `std::vector` on CPU to a `battery::vector` on GPU (notice you don't need to do any manual memory allocation or deallocation):

```cpp
#include <vector>
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
```

## Common Questions

* [How to transfer data from the CPU to the GPU?](https://lattice-land.github.io/2-cuda-battery.html)
* [How to create a CMake project for CUDA project?](https://lattice-land.github.io/3-cuda-battery.html)
* [How to allocate a vector shared by all threads of a block inside a kernel?](https://lattice-land.github.io/4-cuda-battery.html#block-local-memory)
* [How to allocate a vector shared by all blocks inside a kernel?](https://lattice-land.github.io/4-cuda-battery.html#grid-local-memory)
* [CUDA runtime error an illegal memory access was encountered](https://lattice-land.github.io/4-cuda-battery.html#avoiding-obscure-cuda-runtime-errors)
* [How to allocate a vector in shared memory?](https://lattice-land.github.io/5-cuda-battery.html)

## Quick Reference

* Namespace: `battery::*`.
* The documentation is not exhaustive (which is why we provide a link to the standard C++ STL documentation), but we document most of the main differences and the features without a standard counterpart.
* The table below is a quick reference to the most useful features, but it is not exhaustive.
* _The structures provided here are not thread-safe, this responsibility is delegated to the user of this library._

| Category | Main features | | | |
| --- | --- | --- | --- | --- |
| [Allocator](https://lattice-land.github.io/cuda-battery/allocator_8hpp.html)  | [`standard_allocator`](https://lattice-land.github.io/cuda-battery/classbattery_1_1standard__allocator.html) | [`global_allocator`](https://lattice-land.github.io/cuda-battery/classbattery_1_1global__allocator.html) | [`managed_allocator`](https://lattice-land.github.io/cuda-battery/classbattery_1_1managed__allocator.html) | [`pool_allocator`](https://lattice-land.github.io/cuda-battery/classbattery_1_1pool__allocator.html) |
| Pointers | [`shared_ptr`](https://lattice-land.github.io/cuda-battery/classbattery_1_1shared__ptr.html) ([`std`](https://en.cppreference.com/w/cpp/memory/shared_ptr)) | [`make_shared`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a51fbd102c2aef01adc744cee4bc35ea9) ([`std`](https://en.cppreference.com/w/cpp/memory/shared_ptr/make_shared)) | [`allocate_shared`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#aee9241d882f78f0130435b46389ff9ac) ([`std`](https://en.cppreference.com/w/cpp/memory/shared_ptr/allocate_shared)) | |
| | [`unique_ptr`](https://lattice-land.github.io/cuda-battery/classbattery_1_1unique__ptr.html) ([`std`](https://en.cppreference.com/w/cpp/memory/unique_ptr)) | [`make_unique`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#aa354aeb0995d495b2b9858a6ea2fb568) ([`std`](https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique)) | [`make_unique_block`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#aa781d8577b63b6e7b789223e825f52c3) | [`make_unique_grid`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a5ba4adb6d7953ca7e8c6602b7e455b14) |
| Containers | [`vector`](https://lattice-land.github.io/cuda-battery/vector_8hpp.html) ([`std`](https://en.cppreference.com/w/cpp/container/vector)) | [`string`](https://lattice-land.github.io/cuda-battery/string_8hpp.html) ([`std`](https://en.cppreference.com/w/cpp/string/basic_string)) | [`dynamic_bitset`](https://lattice-land.github.io/cuda-battery/dynamic__bitset_8hpp.html) | |
| | [`tuple`](https://en.cppreference.com/w/cpp/utility/tuple) | [`variant`](https://lattice-land.github.io/cuda-battery/variant_8hpp.html) ([`std`](https://en.cppreference.com/w/cpp/utility/variant)) | [`bitset`](https://lattice-land.github.io/cuda-battery/bitset_8hpp.html) ([`std`](https://en.cppreference.com/w/cpp/utility/bitset)) | |
| Utility | [`CUDA`](https://lattice-land.github.io/cuda-battery/utility_8hpp.html#a7b01e29f669d6beed251f1b2a547ca93) | [`INLINE`](https://lattice-land.github.io/cuda-battery/utility_8hpp.html#a2eb6f9e0395b47b8d5e3eeae4fe0c116) | [`CUDAE`](https://lattice-land.github.io/cuda-battery/utility_8hpp.html#a289596c1db721f82251de6902f9699db) | [`CUDAEX`](https://lattice-land.github.io/cuda-battery/utility_8hpp.html#af35c92d967acfadd086658422f631100) |
| | [`limits`](https://lattice-land.github.io/cuda-battery/structbattery_1_1limits.html) | [`ru_cast`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a7fdea425c76eab201a009ea09b8cbac0) | [`rd_cast`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#aa2296c962277e71780bccf1ba9708f59) | |
| | [`popcount`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a2821ae67e8ea81b375f3fd6d70909fef) ([`std`](https://en.cppreference.com/w/cpp/numeric/popcount)) | [`countl_zero`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#aa18a34122dc3e8b7e96c4a54eeffa9aa) ([`std`](https://en.cppreference.com/w/cpp/numeric/countl_zero)) | [`countl_one`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#ae252a50e577d7b092eb368b7e0289772) ([`std`](https://en.cppreference.com/w/cpp/numeric/countl_one)) | [`countr_zero`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a7338f90fab224e49c3716c5eace58bee) ([`std`](https://en.cppreference.com/w/cpp/numeric/countr_zero)) |
| | [`countr_one`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a974d0a682d546e1185ae7dca29c272d6) ([`std`](https://en.cppreference.com/w/cpp/numeric/countr_one)) | [`signum`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a31b3f5ee3799a73d29c153ebd222cdea) | [`ipow`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a93472d80842253624e2436eef7b900b6) | |
| | [`add_up`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#af3a4582a08267940dbdb5b39044aa4c6) | [`add_down`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a43d013f1db8f8b8c085c544859e24a7f) | [`sub_up`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a6d6340503a20225d569990c0044519bb) | [`sub_down`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a32ff1fe9f8d2eac8fd7a2d08b0110461) |
| | [`mul_up`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#ae3edf2725aaea683aff7a100733b26f2) | [`mul_down`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a6dd3e5546b5286d98cb29c7560542759) | [`div_up`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a3ce4b4df0f80c5b19c7d3b401464f309) | [`div_down`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#ac253a56f7fa54ade8f2eb762d3b317f9) |
| [Memory](https://lattice-land.github.io/cuda-battery/memory_8hpp.html)  | [`local_memory`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a09111ca968cc4d8defa60555963dd052) | [`read_only_memory`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a22ff3da8ce553868de9c2b8fe604fe3c) | [`atomic_memory`](https://lattice-land.github.io/cuda-battery/classbattery_1_1atomic__memory.html) | |
| | [`atomic_scoped_memory`](https://lattice-land.github.io/cuda-battery/classbattery_1_1atomic__memory__scoped.html) | [`atomic_memory_block`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#afb485d8f961537d1ca590f78d16ac1c4) | [`atomic_memory_grid`](https://lattice-land.github.io/cuda-battery/namespacebattery.html#a2af42ce969d94b6b8bb1ed9a94b9cf49) | |
