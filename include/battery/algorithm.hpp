// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef CUDA_BATTERY_ALGORITHM_HPP
#define CUDA_BATTERY_ALGORITHM_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "tuple.hpp"

namespace battery {

namespace impl {

template <class Seq, class Compare>
CUDA int partition(Seq& seq, int low, int high, Compare comp) {
  int i = low - 1; // Index of smaller element

  for (int j = low; j <= high - 1; j++) {
    // If current element is smaller than the pivot
    if (comp(j, high)) {
      i++; // Increment index of smaller element
      ::battery::swap(seq[i], seq[j]);
    }
  }
  ::battery::swap(seq[i + 1], seq[high]);
  return i + 1;
}

template <class Seq, class Compare>
CUDA void quickSort(Seq& seq, Compare comp) {
  if(seq.size() < 2) {
    return;
  }
  vector<int> stackl;
  vector<int> stackh;
  stackl.push_back(0);
  stackh.push_back(seq.size() - 1);
  while(stackl.size() > 0) {
    int low = stackl.back(); stackl.pop_back();
    int high = stackh.back(); stackh.pop_back();
    if(low < high) {
      // pi is partitioning index, seq[p] is now at right place
      int pi = partition(seq, low, high, comp);
      // Separately sort elements before partition and after partition
      // quickSort(seq, low, pi - 1, comp);
      stackl.push_back(low);
      stackh.push_back(pi-1);
      // quickSort(seq, pi + 1, high, comp);
      stackl.push_back(pi+1);
      stackh.push_back(high);
    }
  }
}
}// namespace impl

/** Sort the sequence `seq` in-place.
 * The underlying algorithm is an iterative version of quicksort.
 * * `comp(a, b)` returns `true` whenever a < b, and false otherwise. */
template <class Seq, class Compare>
CUDA NI void sort(Seq& seq, Compare comp) {
  assert(seq.size() < limits<int>::top());
  impl::quickSort(seq, [&](int i, int j) { return comp(seq[i], seq[j]); });
}

/** Similar to `sort`, but the comparison function is takes indexes instead of elements themselves.
 * * `comp(i, j)` returns `true` whenever seq[i] < seq[j], and false otherwise. */
template <class Seq, class Compare>
CUDA NI void sorti(Seq& seq, Compare comp) {
  assert(seq.size() < limits<int>::top());
  impl::quickSort(seq, comp);
}

} // namespace battery

#endif
