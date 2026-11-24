// Copyright 2026 Pierre Talbot

#ifndef CUDA_BATTERY_GROUP_HPP
#define CUDA_BATTERY_GROUP_HPP

#include "utility.hpp"

#ifdef __CUDACC__
  #include <cooperative_groups.h>
  #include <cooperative_groups/reduce.h>
#endif

/** \file group.hpp
 * Groups of threads executing a collective operation together, with the same interface for the current thread, a warp, a block, the warps of a block and a grid.
 *
 * A group provides `rank()`, `size()`, `bool any_sync(bool p)`, a barrier returning the logical or of `p` over all its threads, and `sync()`, a plain barrier.
 * For instance, the loop `for(int k = g.rank(); k < n; k += g.size())` splits the iterations `0, ..., n-1` among the members of the group `g`.
 * | Group         | Threads                              | `any_sync`                           |
 * |----------------|--------------------------------------|--------------------------------------|
 * | `Sequential`   | the current thread                   | none                                 |
 * | `ThisBlock`    | the threads of the block             | `__syncthreads_or`                   |
 * | `WarpsOfBlock` | the warps of the block               | `__syncthreads_or`                   |
 * | `ThisWarp`     | the lanes of the warp                | `__any_sync`                         |
 * | `GridGroup`    | the threads of the grid              | `__syncthreads_or` and `grid.sync()` |
 *
 * Synchronization convention for the collective methods of an object, i.e., the methods called by all the threads of a group:
 * A collective method does not synchronize the threads when it returns: add a barrier before reading its results.
 * It does not synchronize on entry either: the caller guarantees that all the threads have exited the other methods of the same object (e.g., with a barrier) before calling it.
 */

namespace battery {

/** A group made of the current thread only. */
struct Sequential {
  CUDA INLINE constexpr int rank() const { return 0; }
  CUDA INLINE constexpr int size() const { return 1; }
  CUDA INLINE constexpr bool any_sync(bool p) const { return p; }
  CUDA INLINE constexpr void sync() const {}
};

#ifdef __CUDACC__

/** All the threads of a (one-dimensional) block. */
struct ThisBlock {
  __device__ INLINE int rank() const { return threadIdx.x; }
  __device__ INLINE int size() const { return blockDim.x; }
  __device__ INLINE bool any_sync(bool p) const { return __syncthreads_or(p); }
  __device__ INLINE void sync() const { __syncthreads(); }

  /** A barrier returning the number of threads of the block for which `p` is `true`. */
  __device__ INLINE int count_sync(bool p) const { return __syncthreads_count(p); }
};

/** All the warps of a (one-dimensional) block: the rank of a thread is the index of its warp.
 * In the loop `for(int k = rank(); k < n; k += size())`, the 32 lanes of the warp `w` execute the iterations `k = w, w + num_warps, ...` together.
 * Hence, the body of the loop can use warp-level primitives, such as the methods of `ThisWarp`.
 * `blockDim.x` must be a multiple of 32. */
struct WarpsOfBlock {
  __device__ INLINE int rank() const { return threadIdx.x / 32; }
  __device__ INLINE int size() const { return blockDim.x / 32; }
  __device__ INLINE bool any_sync(bool p) const { return __syncthreads_or(p); }
  __device__ INLINE void sync() const { __syncthreads(); }
};

/** All the lanes of the warp of the current thread.
 * All the 32 lanes must call these methods together (no divergence). */
struct ThisWarp {
  __device__ INLINE int rank() const { return threadIdx.x % 32; }
  __device__ INLINE int size() const { return 32; }
  __device__ INLINE bool any_sync(bool p) const { return __any_sync(0xffffffffu, p); }
  __device__ INLINE void sync() const { __syncwarp(); }

  /** \return A mask whose bit `r` is the value of `p` in the lane `r`. */
  __device__ INLINE unsigned int ballot_sync(bool p) const { return __ballot_sync(0xffffffffu, p); }

  /** \return The sum of `x` over the lanes of the warp.
   * It uses the `redux` instruction on sm_80 and later, and shuffles before. */
  __device__ INLINE int reduce_add_sync(int x) const {
    namespace cg = cooperative_groups;
    return cg::reduce(cg::tiled_partition<32>(cg::this_thread_block()), x, cg::plus<int>());
  }
};

/** All the threads of the grid, the kernel must be launched with `cudaLaunchCooperativeKernel`.
 * Allocate the group in global memory and share it among the blocks (e.g., with `battery::make_unique_grid`, or in managed memory).
 * It holds no data specific to a launch, so it can be reused across kernel launches.
 *
 * There is no reduction over a grid in cooperative groups (`cooperative_groups::reduce` only supports tiles), hence `any_sync` reduces in three shared flags, `any_true`, used in rotation.
 * It costs one `grid.sync()` per call, and consecutive loops calling it do not need a barrier in between. */
class GridGroup {
  /** `any_true[t % 3]` is `true` if a thread called `any_sync(true)` in the call `t`, where `t` counts the calls of `any_sync` since the construction.
   * We do not use atomics because tearing is seemingly not possible in CUDA (according to information given by Nvidia engineers during a hackathon). */
  bool any_true[3];

  /** `t % 3`, the flag of the current call, see `any_sync`. */
  int slot;

public:
  CUDA GridGroup(): any_true{false, false, false}, slot(0) {}

  /** `this_grid()` reads two special registers (no memory load), and its result is specific to the launch, hence we do not store it. */
  __device__ INLINE int rank() const { return static_cast<int>(cooperative_groups::this_grid().thread_rank()); }
  __device__ INLINE int size() const { return static_cast<int>(cooperative_groups::this_grid().num_threads()); }
  __device__ INLINE void sync() const { cooperative_groups::this_grid().sync(); }

  /** The call `t` writes and reads `any_true[t % 3]`, so a thread reading the flag of the call `t` is not disturbed by a fast thread already in the call `t + 1`.
   * The flag of the call `t + 1` is reset during the call `t`, after all the threads read it in the call `t - 2`. */
  __device__ bool any_sync(bool p) {
    // It also makes the write of `slot` by the thread 0 of this block, in the previous call, visible to the threads of this block.
    bool block_any = __syncthreads_or(p);
    int s = slot;
    int next = s == 2 ? 0 : s + 1;
    if(block_any && threadIdx.x == 0) {
      any_true[s] = true;
    }
    // `any_true[next]` is written in the next call, and was last read two calls ago.
    if(rank() == 0) {
      any_true[next] = false;
    }
    sync();
    bool cont = any_true[s];
    // All the threads read `slot` in this call before `sync()`, and they read it again in the next call after `__syncthreads_or`.
    // Every block writes the same value.
    if(threadIdx.x == 0) {
      slot = next;
    }
    return cont;
  }
};

#endif // __CUDACC__

} // namespace battery

#endif
