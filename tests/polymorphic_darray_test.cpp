// Copyright 2021 Pierre Talbot

#include "darray.hpp"
#include "allocator.hpp"

using namespace battery;

class A;
class B;
CUDA_GLOBAL void init_A(A** p, int uid);
CUDA_GLOBAL void init_B(B** p, int uid);

class A {
public:
  int uid;
  CUDA A(int i): uid(i){}
  CUDA virtual int id() { return uid; }
  virtual A* clone(StandardAllocator& alloc) const {
    return new(alloc) A(uid);
  }
  #if __NVCC__
    CUDA virtual A* clone(GlobalAllocatorGPU& alloc) const {
      return new(alloc) A(uid);
    }

    virtual A* clone(GlobalAllocatorCPU& alloc) const {
      ManagedAllocator managed_allocator;
      A** a = new(managed_allocator) A*;
      init_A<<<1, 1>>>(a, uid);
      CUDIE(cudaDeviceSynchronize());
      A* ap = *a;
      managed_allocator.deallocate(a);
      return ap;
    }
  #endif
};

class B: public A {
public:
  CUDA B(int i): A(i) {}
  CUDA virtual int id() { return uid * 10; }
  virtual B* clone(StandardAllocator& alloc) const {
    return new(alloc) B(uid);
  }
  #if __NVCC__
    CUDA virtual B* clone(GlobalAllocatorGPU& alloc) const {
      return new(alloc) B(uid);
    }

    virtual B* clone(GlobalAllocatorCPU& alloc) const {
      ManagedAllocator managed_allocator;
      B** b = new(managed_allocator) B*;
      init_B<<<1, 1>>>(b, uid);
      CUDIE(cudaDeviceSynchronize());
      B* bp = *b;
      managed_allocator.deallocate(b);
      return bp;
    }
  #endif
};

CUDA_GLOBAL void init_A(A** a, int uid) {
  *a = new A(uid);
}

CUDA_GLOBAL void init_B(B** b, int uid) {
  *b = new B(uid);
}

__device__ int test_poly_array(DArray<A*, GlobalAllocatorGPU>& poly) {
  if(poly.size() != 3) { printf("expected size of 3."); return 1; }
  int uids[3] = {1, 20, 30};
  for(int i = 0; i < poly.size(); ++i) {
    if(poly[i]->id() != uids[i]) {
      printf("expected poly[%d]->id() = %d, but got %d\n", i, uids[i], poly[i]->id());
      return 1;
    }
  }
  return 0;
}

CUDA_GLOBAL void test_poly(int* res, DArray<A*, GlobalAllocatorCPU>* poly_ptr) {
  // Test the polymorphic copy that was done on host side.
  DArray<A*, GlobalAllocatorGPU> poly = poly_ptr->viewOnGPU();
  *res = test_poly_array(poly);
  if(*res == 1) { printf("Testing poly failed\n"); return; }
  // Test copy on GPU.
  DArray<A*, GlobalAllocatorGPU> poly2(poly);
  *res = test_poly_array(poly2);
  if(*res == 1) { printf("Testing poly2 failed\n"); return; }
}

// This test the variant class on GPU, it could be extracted in another class, but for now, let's keep the cmake file simple.
// We also test how it works in cooperation with array so I guess it's ok to leave it here for now.
#include "variant.hpp"
CUDA_GLOBAL void test_variant(int* res) {
  using Arr = DArray<int, GlobalAllocatorGPU>;
  using DataT = variant<char, Arr>;
  DataT a1(DataT::create<1>(Arr(3, 2)));
  DataT a2(DataT::create<1>(Arr(3, 2)));
  DataT a3(DataT::create<0>(100));
  if(a1 != a2) { *res = 1; printf("Testing a1 == a2 failed\n"); return; }
  if(a1 == a3) { *res = 1; printf("Testing a1 != a3 failed\n"); return; }
  if(a1.index() != 1) { *res = 1; printf("Testing a1.index() != 1 failed\n"); return; }
  if(a3.index() != 0) { *res = 1; printf("Testing a3.index() != 0 failed\n"); return; }
  if(get<0>(a3) != 100) { *res = 1; printf("Testing get<0>(a3) != 100 failed\n"); return; }
  if(get<1>(a1) != Arr(3, 2)) { *res = 1; printf("Testing get<1>(a3) != a2 failed\n"); return; }
}

int main() {
  StandardAllocator standard_allocator;
  ManagedAllocator managed_allocator;
  DArray<A*, GlobalAllocatorCPU> test(3);
  DArray<A*, StandardAllocator> cpu_poly(3);
  cpu_poly[0] = new(standard_allocator) A(1);
  cpu_poly[1] = new(standard_allocator) B(2);
  cpu_poly[2] = new(standard_allocator) B(3);
  // We copy `cpu_poly` using a `GlobalAllocator` which should triggers the polymorphic `clone` method.
  auto gpu_poly = new(managed_allocator) DArray<A*, GlobalAllocatorCPU>(cpu_poly);
  int* res = new(managed_allocator) int(0);
  test_poly<<<1,1>>>(res, gpu_poly);
  CUDIE(cudaDeviceSynchronize());
  if(*res == 0) {
    test_variant<<<1,1>>>(res);
    CUDIE(cudaDeviceSynchronize());
  }
  return *res;
}
