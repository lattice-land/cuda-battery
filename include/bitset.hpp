// Copyright 2021 Pierre Talbot, Frédéric Pinel, Cem Guvel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BITSET_HPP
#define BITSET_HPP

#include <cstdio>
#include <cassert>
#include "cuda_helper.hpp"

template <size_t N>
struct Bitset {
  int set[N];

  const int iSize = sizeof(int) * CHAR_BIT;

  // Constructors
  CUDA Bitset (): set(){}

  //CUDA ~Bitset (){}

  // v copy constructor
  CUDA Bitset(const Bitset<N>& other): set() {
    memcpy(set, other.set, N);
  }

  // v same as copy (for most part)
  CUDA Bitset<N>& operator = (const Bitset<N>& other) {
    if (this != &other) {
      memcpy(this, &other, sizeof(Bitset<N>));
    }
    return *this;
  }
  // <<< 

  // Static Functions
  CUDA static Bitset<N> universe () {
    return Bitset<N>().take_complement();
  }

  CUDA static Bitset<N> empty () {
    return Bitset<N>();
  }
  // <<< 

  // Set R Element Operations
  CUDA void add (int x) {
    int bound = iSize * N;
    int two = (x + bound) % bound;
    int bit_index = two % iSize;
    int row_index = two / iSize;
    set[row_index] |= 1 << bit_index;
  }

  CUDA void remove (int x) {
    int bound = iSize * N;
    int two = (x + bound) % bound;
    int bit_index = two % iSize;
    int row_index = two / iSize;
    set[row_index] &= ~(1 << bit_index);
  }

  CUDA bool contains (int x) const {
    int bound = iSize * N;
    int two = (x + bound) % bound;
    int bit_index = two % iSize;
    int row_index = two / iSize;
    return set[row_index] & (1 << bit_index);
  }
  // <<< 

  // Binary Operations
  CUDA bool operator == (const Bitset<N>& rhs) const {
    bool result = true;
    for (int i = 0; i < N; ++i) {
      result &= (set[i] == rhs.set[i]);
    }
    return result;
  }

  CUDA bool operator != (const Bitset<N>& rhs) const {
    return !(*this == rhs);
  }

  CUDA bool operator >= (const Bitset<N>& rhs) const {
    bool result = true;
    for (int i = 0; i < N; ++i) {
      result &= (0 == (rhs.set[i] & (rhs.set[i] ^ set[i])));
    }
    return result;
  }

  CUDA bool operator < (const Bitset<N>& rhs) const {
    return !(*this >= rhs);
  }

  CUDA bool operator <= (const Bitset<N>& rhs) const {
    return rhs >= *this;
  }

  CUDA bool operator > (const Bitset<N>& rhs) const {
    return rhs < *this;
  }
  // <<<

  CUDA int size() const {
    int ret = 0;
    for (int i=0; i < N; ++i){ 
      int inter = set[i];
      for(;inter != 0; ++ret){
        inter &= inter - 1;
      }
    }
    return ret;
  }

  CUDA Bitset diff (Bitset const &other) const {
    Bitset res;
    for (int i=0; i < N; ++i) {
      res.set[i] &= ~(other.set[i]);
    }
    return res;
  }

  CUDA Bitset set_union (Bitset const &x) const {
    Bitset ret;
    for(int i=0;i < N; ++i){
      ret.set[i] = x.set[i] | set[i];
    }
    return ret;
  }

  CUDA void union_with (const Bitset<N>& other) {
    for (int i = 0; i < N; ++i) {
      set[i] |= other.set[i];
    }
  }

  CUDA Bitset set_intersection (Bitset const &x) const {
    Bitset ret;
    for(int i=0;i < N; ++i){
      ret.set[i] = x.set[i] & set[i];
    }
    return ret;
  }

  CUDA void intersection_with (const Bitset<N>& other) {
    for (int i = 0; i < N; ++i) {
      set[i] &= other.set[i];
    }
  }

  CUDA void take_complement() {
    for (int i = 0; i < N; ++i) {
      set[i] = ~set[i];
    }
  }

  CUDA Bitset<N> complement() {
    Bitset<N> result;
    result.take_complement();
    return result;
  }

  CUDA void print() const {
    for (int j = 0;j < N; ++j){
      printf("%4d: ", j);
      for (int i = 0; i < iSize; i++){
        if (!(i % 8)) { printf("|"); }
        printf("%d",1 & (set[j] >> i));
      }
      printf("|\n");
    }
  }

  CUDA int max() const {
    int rel_i = N - 1;
    for (int j=0; j < N; ++j) {
      if (rel_i == -1) { rel_i = N-1;}
      // Maybe for CPU
      //if (set[rel_i] == 0) { --rel_i; continue; }
      for (int s=1; s <= iSize; ++s) {
        if(set[rel_i] & (1 << (iSize - s))) {
          return iSize * (N - j) - s;
        }
      }
      --rel_i;
    }
    return -N * iSize - 1;
  }

  CUDA int min() const {
    int rel_i = N;
    for (int j=0; j < N ;++j) {
      if (rel_i == N) { rel_i = 0; }
      for (int bit_i=0; bit_i < iSize;++bit_i) {
        if (set[rel_i] & (1 << bit_i)) {
          return iSize * (j - N) + bit_i;
        }
      }
      ++rel_i;
    }
    return N * iSize;
  }
};

#endif
