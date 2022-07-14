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
#include "utility.hpp"
#include "string.hpp"

namespace battery {

template <size_t N, class Mem, class T = unsigned long long>
class Bitset {
private:
  constexpr static size_t BITS_PER_BLOCK = sizeof(T) * CHAR_BIT;
  constexpr static size_t BLOCKS = N / BITS_PER_BLOCK + (N % BITS_PER_BLOCK != 0);
  constexpr static size_t MAX_SIZE = N;
  constexpr static T ZERO = T{0};
  constexpr static T ONES = ~T{0};

  using block_type = typename Mem::atomic_type<T>;

  /** Suppose T = char, with 2 blocks. Then the bitset is represented as:
   *
   *    blocks index:       1       0
   *    blocks:         00000000 00100000
   *    indexes:           ...98 76543210
   *
   *    Hence, `bitset.test(5) == true`.
   *  */
  block_type blocks[BLOCKS];

public:
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "Block of a bitset must be defined on an unsigned integer type.");

  CUDA Bitset(): blocks() {}

  CUDA Bitset(const char* bit_str): blocks() {
    size_t n = min(strlen(bit_str), N);
    for(int i = n-1; i >= 0; i--) {
      if(bit_str[i] == '1') {
        set(i);
      }
    }
  }

  template<class Mem2>
  CUDA Bitset(const Bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      Mem::store(blocks[i], Mem::load(other[i]));
    }
  }

  CUDA static Bitset zeroes() {
    return Bitset();
  }

  CUDA static Bitset ones() {
    return Bitset().set();
  }

private:
  CUDA block_type& block_of(size_t pos) {
    return blocks[pos / BITS_PER_BLOCK];
  }

  CUDA T load_block(size_t pos) const {
    return Mem::load(block_of(pos));
  }

  CUDA void store_block(size_t pos, T block) {
    Mem::store(block_of(pos), block);
  }

  CUDA size_t bit_of(size_t pos) {
    return 1 << (pos % BITS_PER_BLOCK);
  }

public:
  CUDA bool test(size_t pos) const {
    assert(pos < MAX_SIZE);
    return load_block(pos) & bit_of(pos);
  }

  CUDA bool all() const {
    for(int i = 0; i < BLOCKS; ++i) {
      if(Mem::load(blocks[i]) != ONES) {
        return false;
      }
    }
    return true;
  }

  CUDA bool any() const {
    for(int i = 0; i < BLOCKS; ++i) {
      if(Mem::load(blocks[i]) > ZERO) {
        return true;
      }
    }
    return false;
  }

  CUDA bool none() const {
    for(int i = 0; i < BLOCKS; ++i) {
      if(Mem::load(blocks[i]) != ZERO) {
        return false;
      }
    }
    return true;
  }

  CUDA size_t size() const {
    return MAX_SIZE;
  }

  CUDA size_t count() const {
    size_t bits_at_one = 0;
    for(int i = 0; i < BLOCKS; ++i){
      bits_at_one += popcount(blocks[i]);
    }
    return bits_at_one;
  }

  CUDA Bitset& set() {
    for(int i = 0; i < BLOCKS; ++i) {
      Mem::store(blocks[i], ONES);
    }
    return *this;
  }

  CUDA Bitset& set(size_t pos) {
    assert(pos < MAX_SIZE);
    store_block(pos, load_block(pos) | bit_of(pos));
    return *this;
  }

  CUDA Bitset& set(size_t pos, bool value) {
    assert(pos < MAX_SIZE);
    return value ? set(pos) : reset(pos);
  }

  CUDA Bitset& reset() {
    for(int i = 0; i < BLOCKS; ++i) {
      Mem::store(blocks[i], ZERO);
    }
    return *this;
  }

  CUDA Bitset& reset(size_t pos) {
    assert(pos < MAX_SIZE);
    store_block(pos, load_block(pos) & ~bit_of(pos));
  }

  CUDA Bitset& flip() {
    for(int i = 0; i < BLOCKS; ++i) {
      Mem::store(blocks[i], ~Mem::load(blocks[i]));
    }
  }

  CUDA Bitset& flip(size_t pos) {
    assert(pos < MAX_SIZE);
    store_block(pos, load_block(pos) ^ bit_of(pos));
  }

  template<class Mem2>
  CUDA Bitset& operator&=(const Bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      Mem::store(blocks[i], Mem::load(blocks[i]) & Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  template<class Mem2>
  CUDA Bitset& operator|=(const Bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      Mem::store(blocks[i], Mem::load(blocks[i]) | Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  template<class Mem2>
  CUDA Bitset& operator^=(const Bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      Mem::store(blocks[i], Mem::load(blocks[i]) ^ Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  CUDA Bitset operator~() const {
    return Bitset(*this).flip();
  }

  template<class Mem2>
  CUDA bool operator==(const Bitset<N, Mem2, T>& other) const {
    for(int i = 0; i < BLOCKS; ++i) {
      if(blocks[i] != other.blocks[i]) {
        return false;
      }
    }
    return true;
  }

  template<class Mem2>
  CUDA bool operator!=(const Bitset<N, Mem2, T>& other) const {
    return !(*this == other);
  }

  CUDA int countl_zero() const {
    int k = BLOCKS - 1;
    for(; k >= 0 && blocks[k] == ZERO; --k) {}
    return (BLOCKS - 1 - k) * BITS_PER_BLOCK + (k == -1 ? 0 : ::battery::countl_zero(blocks[k]));
  }

  CUDA int countl_one() const {
    int k = BLOCKS - 1;
    for(; k >= 0 && blocks[k] == ONES; --k) {}
    return (BLOCKS - 1 - k) * BITS_PER_BLOCK + (k == -1 ? 0 : ::battery::countl_one(blocks[k]));
  }

  CUDA int countr_zero() const {
    int k = 0;
    for(; k < BLOCKS && blocks[k] == ZERO; ++k) {}
    return k * BITS_PER_BLOCK + (k == BLOCKS ? 0 : ::battery::countr_zero(blocks[k]));
  }

  CUDA int countr_one() const {
    int k = 0;
    for(; k < BLOCKS && blocks[k] == ONES; ++k) {}
    return k * BITS_PER_BLOCK + (k == BLOCKS ? 0 : ::battery::countr_one(blocks[k]));
  }

  CUDA void print() const {
    for(int i = size() - 1; i >= 0; --i) {
      printf("%d", 1 & test(i));
    }
  }

  template <class Allocator>
  CUDA String<Allocator> to_string(Allocator allocator = Allocator()) const {
    String<Allocator> bits_str(size(), allocator);
    for(int i = size() - 1, j = 0; i >= 0; --i, ++j) {
      bits_str[j] = test(i) ? '1' : '0';
    }
    return bits_str;
  }
};

template<size_t N, class Mem, class T>
Bitset<N, Mem, T> operator&(const Bitset<N, Mem, T>& lhs, const Bitset<N, Mem, T>& rhs) {
  return Bitset<N, Mem, T>(lhs) &= rhs;
}

template<size_t N, class Mem, class T>
Bitset<N, Mem, T> operator|(const Bitset<N, Mem, T>& lhs, const Bitset<N, Mem, T>& rhs) {
  return Bitset<N, Mem, T>(lhs) |= rhs;
}

template<size_t N, class Mem, class T>
Bitset<N, Mem, T> operator^(const Bitset<N, Mem, T>& lhs, const Bitset<N, Mem, T>& rhs) {
  return Bitset<N, Mem, T>(lhs) ^= rhs;
}

} // namespace battery

#endif
