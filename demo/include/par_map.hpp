// Copyright 2023 Pierre Talbot

#ifndef DEMO_PAR_MAP_HPP
#define DEMO_PAR_MAP_HPP

#include <functional>
#include <span>
#include "battery/utility.hpp"

/** Given a sequence `seq` and a number of blocks, returns the size of a chunk that each block must process. */
CUDA size_t chunk_size_per_block(const auto& seq, size_t num_blocks) {
  return seq.size() / num_blocks + (seq.size() % num_blocks != 0);
}

/** Given a sequence `seq`, returns a view of the sequence starting at `from` and finishing at `min(seq.size(), from + size)` (thus correctly handling the case of the last block possibly having a chunk too large).
 *
 * NOTE: We can use `std::span` in `device` code because we only used `constexpr` methods of this class (enabled with the flag --expt-relaxed-constexpr). */
template <class Sequence>
CUDA std::span<typename Sequence::value_type> make_safe_span(Sequence& seq, size_t from, size_t size) {
  if(from >= seq.size()) {
    return std::span<typename Sequence::value_type>();
  }
  else {
    return std::span<typename Sequence::value_type>(seq.data() + from, min(seq.size() - from, size));
  }
}

/** A block-parallel map operation.
 *   - On CPU, can be run with `block_par_map(v, f)`
 *   - On GPU, can be run with `block_par_map(v, f, blockDim.x, threadIdx.x)`
 */
template <class Sequence, class F>
CUDA void block_par_map(Sequence& v, F&& f, size_t stride_size = 1, size_t offset = 0) {
  for(size_t i = offset; i < v.size(); i += stride_size) {
    v[i] = f(v[i]); // std::invoke(std::forward<F>(f), v[i]);
  }
}

/** A grid-parallel map operation, each block processes a chunk of the vector `v`. */
template <class Sequence, class F>
__device__ void grid_par_map(Sequence& v, F&& f) {
  size_t slice_size = chunk_size_per_block(v, gridDim.x);
  auto chunk = make_safe_span(v, slice_size * blockIdx.x, slice_size);
  block_par_map(chunk, std::forward<F>(f), blockDim.x, threadIdx.x);
}

#endif
