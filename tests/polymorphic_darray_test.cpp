// Copyright 2021 Pierre Talbot

#include "allocator.hpp"
#include "darray.hpp"

class A;
class B;
CUDA_GLOBAL void init_A(A** p, int uid);
CUDA_GLOBAL void init_B(B** p, int uid);

class A {
public:
  int uid;
  CUDA A(int i): uid(i){}
  CUDA virtual int id() { return uid; }
  virtual A* clone(StandardAllocator&) const {
    return new(standard_allocator) A(uid);
  }
  #if __NVCC__
    CUDA virtual A* clone(GlobalAllocator& global_allocator) const {
      #if ON_GPU()
        return new(global_allocator) A(uid);
      #else
        A** a;
        CUDIE(cudaMallocManaged(&a, sizeof(A*)));
        init_A<<<1, 1>>>(a, uid);
        CUDIE(cudaDeviceSynchronize());
        A* ap = *a;
        cudaFree(a);
        return ap;
      #endif
    }
  #endif
};

class B: public A {
public:
  CUDA B(int i): A(i) {}
  CUDA virtual int id() { return uid * 10; }
  virtual B* clone(StandardAllocator& standard_allocator) const {
    return new(standard_allocator) B(uid);
  }
  #if __NVCC__
    CUDA virtual B* clone(GlobalAllocator& global_allocator) const {
      #if ON_GPU()
        return new(global_allocator) B(uid);
      #else
        B** b;
        CUDIE(cudaMallocManaged(&b, sizeof(B*)));
        init_B<<<1, 1>>>(b, uid);
        CUDIE(cudaDeviceSynchronize());
        B* bp = *b;
        cudaFree(b);
        return bp;
      #endif
    }
  #endif
};

CUDA_GLOBAL void init_A(A** a, int uid) {
  *a = new A(uid);
}

CUDA_GLOBAL void init_B(B** b, int uid) {
  *b = new B(uid);
}

__device__ int test_poly_array(DArray<A*, GlobalAllocator>& poly) {
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

CUDA_GLOBAL void test_poly(int* res, DArray<A*, GlobalAllocator>* poly_ptr) {
  // Test the polymorphic copy that was done on host side.
  DArray<A*, GlobalAllocator>& poly = *poly_ptr;
  printf("Testing poly starts...\n");
  *res = test_poly_array(poly);
  printf("Testing poly ends...\n");
  if(*res == 1) { printf("Testing poly failed\n"); return; }
  // Test copy on GPU.
  DArray<A*, GlobalAllocator> poly2(poly);
  *res = test_poly_array(poly2);
  if(*res == 1) { printf("Testing poly2 failed\n"); return; }
}

int main() {
  DArray<A*, StandardAllocator> cpu_poly(3);
  cpu_poly[0] = new A(1);
  cpu_poly[1] = new B(2);
  cpu_poly[2] = new B(3);
  // We copy `cpu_poly` using a `GlobalAllocator` which should triggers the polymorphic `clone` method.
  auto gpu_poly = new(managed_allocator) DArray<A*, GlobalAllocator>(cpu_poly);
  int* res = new(managed_allocator) int(0);
  test_poly<<<1,1>>>(res, gpu_poly);
  CUDIE(cudaDeviceSynchronize());
  return *res;
}
