// Copyright 2023 Pierre Talbot

#include <vector>
#include <string>
#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/allocator.hpp"

#include "par_map.hpp"

/** Different aliases to `vector` with different allocators. */
using mvector = battery::vector<int, battery::managed_allocator>;
using gvector = battery::vector<int, battery::global_allocator>;
using pvector = battery::vector<int, battery::pool_allocator>;

/** Example of a CUDA kernel applying a map function to a vector in managed memory. */
__global__ void map_kernel(mvector* v_ptr) {
  grid_par_map(*v_ptr, [](int x){ return x * 2; });
}

/** Similar to `map_kernel`, but first move chunks of the vector to shared_memory. */
__global__ void map_kernel_shared(mvector* v_ptr, size_t shared_mem_capacity) {
  // I. Create a pool of shared memory.
  extern __shared__ unsigned char shared_mem[];
  battery::unique_ptr<battery::pool_allocator, battery::global_allocator> pool_ptr;
  // /!\ We must take a reference to the pool_allocator to avoid copying it, because its copy-constructor is not thread-safe! (It can only be used by one thread at a time).
  battery::pool_allocator& shared_mem_pool = battery::make_unique_block(pool_ptr, static_cast<unsigned char*>(shared_mem), shared_mem_capacity);

  // II. Transfer from global memory to shared memory.
  battery::unique_ptr<pvector, battery::global_allocator> shared_vector;
  size_t chunk_size = chunk_size_per_block(*v_ptr, gridDim.x);
  auto span = make_safe_span(*v_ptr, chunk_size * blockIdx.x, chunk_size);
  pvector& v = battery::make_unique_block(shared_vector, span.data(), span.size(), shared_mem_pool);

  // III. Run the algorithm on the shared memory.
  block_par_map(v, [](int x){ return x * 2; }, blockDim.x, threadIdx.x);
  __syncthreads();

  // IV. Transfer back from shared memory to global memory.
  for(int i = threadIdx.x; i < v.size(); i += blockDim.x) {
    (*v_ptr)[chunk_size * blockIdx.x + i] = v[i];
  }
}

size_t max_shared_mem() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return deviceProp.sharedMemPerBlock;
}

/** Function running a GPU kernel, either with shared memory if the vector `v` is small enough, or in global memory otherwise. */
void map_gpu(std::vector<int>& v, size_t num_blocks) {
  size_t chunk_size_bytes = sizeof(int) * chunk_size_per_block(v, num_blocks);
  auto gpu_v = battery::make_unique<mvector, battery::managed_allocator>(v);
  size_t shared_mem_capacity = max_shared_mem();
  if(shared_mem_capacity > chunk_size_bytes) {
    printf("GPU execution with shared memory (%zu/%zu bytes) and %zu blocks.\n", chunk_size_bytes, shared_mem_capacity, num_blocks);
    map_kernel_shared<<<num_blocks, 256, chunk_size_bytes>>>(gpu_v.get(), chunk_size_bytes);
  }
  else {
    printf("GPU execution with global memory with %zu blocks.\n", num_blocks);
    map_kernel<<<num_blocks, 256>>>(gpu_v.get());
  }
  CUDAEX(cudaDeviceSynchronize());
  // Transfering the new data to the initial vector.
  for(int i = 0; i < v.size(); ++i) {
    v[i] = (*gpu_v)[i];
  }
}

/** Just for comparison with the parallel GPU version. */
void map_cpu(std::vector<int>& v) {
  printf("Computing on CPU.\n");
  block_par_map(v, [](int x) { return x * 2; }); // to demonstrate this function can be called from the CPU as well.
}

int main(int argc, char** argv) {
  if(! ((argc == 3 && argv[1] == std::string("cpu"))
    || (argc == 4 && argv[1] == std::string("gpu"))))
  {
    printf("usage: %s <cpu|gpu> <size> [num-blocks]\n", argv[0]);
    exit(1);
  }
  size_t size = std::stol(argv[2]);
  std::vector<int> original(size, 50);
  std::vector<int> v(original);
  if(argv[1] == std::string("cpu")) {
    map_cpu(v);
  } else {
    size_t num_blocks = std::stol(argv[3]);
    map_gpu(v, num_blocks);
  }
  return 0;
}

// __global__ void local_vector_copy(mvector<int>* v_ptr) {
//   battery::unique_ptr<gvector<int>, battery::global_allocator> block_local;
//   gvector<int>& v_block = battery::make_unique_block(block_local, *v_ptr);
//   // Now each block has its own local copy of the vector `*v_ptr`.
// }
