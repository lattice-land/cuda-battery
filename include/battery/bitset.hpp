// Copyright 2021 Pierre Talbot, Cem Guvel

#ifndef CUDA_BATTERY_BITSET_HPP
#define CUDA_BATTERY_BITSET_HPP

#include <cstdio>
#include <cassert>
#include "utility.hpp"
#include "string.hpp"

/** \file bitset.hpp
 *  An extension of the C++ STL bitset class with support for set operations (e.g., union and intersection) and for atomic read and write on the bitset blocks.
 */

namespace battery {

/**
 * \tparam `N` is the number of bits in the bitset.
 * \tparam `Mem` is a memory consistency, for instance `local_memory`, see `memory.hpp`.
 * \tparam `T` is the underlying type defining the blocks of the bitset.
*/
template <size_t N, class Mem, class T = unsigned long long>
class bitset {
  constexpr static size_t MAX_SIZE = N;
private:
  constexpr static size_t BITS_PER_BLOCK = sizeof(T) * CHAR_BIT;
  constexpr static size_t BITS_LAST_BLOCK = (N % BITS_PER_BLOCK) == 0 ? BITS_PER_BLOCK : (N % BITS_PER_BLOCK);
  constexpr static size_t PADDING_LAST_BLOCK = BITS_PER_BLOCK - BITS_LAST_BLOCK;
  constexpr static size_t BLOCKS = N / BITS_PER_BLOCK + (N % BITS_PER_BLOCK != 0);
  constexpr static T ZERO = T{0};
  constexpr static T ONES = ~T{0};
  /** The last block might not be fully used.
      If we have two blocks of size 8 each, but N = 10, only 2 bits are relevant in the last block.
      We have 10 % 8 = 2, then ONES << 2 gives 11111100, and ~(ONES << 2) = 00000011. */
  constexpr static T ONES_LAST = PADDING_LAST_BLOCK == 0 ? ONES : (T)(~(ONES << BITS_LAST_BLOCK));

  using block_type = typename Mem::template atomic_type<T>;

  /** Suppose T = char, with 2 blocks. Then the bitset "0000 00100000" is represented as:
   *
   *    blocks index:       0       1
   *    blocks:         00100000 00000000
   *    indexes:        76543210    ...98
   *
   *    We have `bitset.test(5) == true`.
   *
   *    Note that the last block is the one carrying the most significant bits, and also the one that is potentially padded with zeroes.
   *  */
  block_type blocks[BLOCKS];

public:
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "Block of a bitset must be defined on an unsigned integer type.");

  CUDA constexpr bitset(): blocks() {}

  CUDA constexpr bitset(const char* bit_str): blocks() {
    size_t n = min(strlen(bit_str), MAX_SIZE);
    for(int i = n, j = 0; i > 0; --i, ++j) {
      if(bit_str[i-1] == '1') {
        set(j);
      }
    }
  }

  template<class Mem2>
  CUDA constexpr bitset(const bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      store(blocks[i], Mem2::load(other[i]));
    }
  }

  CUDA static constexpr bitset zeroes() {
    return bitset();
  }

  CUDA static constexpr bitset ones() {
    return bitset().set();
  }

private:
  CUDA constexpr void store(block_type& block, T data) {
    Mem::store(block, data);
  }

  CUDA constexpr block_type& block_of(size_t pos) {
    return blocks[pos / BITS_PER_BLOCK];
  }

  CUDA constexpr const block_type& block_of(size_t pos) const {
    return blocks[pos / BITS_PER_BLOCK];
  }

  CUDA constexpr T load_block(size_t pos) const {
    return Mem::load(block_of(pos));
  }

  CUDA constexpr void store_block(size_t pos, T data) {
    store(block_of(pos), data);
  }

  CUDA constexpr T bit_of(size_t pos) const {
    return static_cast<T>(1) << (pos % BITS_PER_BLOCK);
  }

public:
  CUDA constexpr bool test(size_t pos) const {
    assert(pos < MAX_SIZE);
    return load_block(pos) & bit_of(pos);
  }

  CUDA constexpr bool all() const {
    int i = 0;
    for(; i < BLOCKS - 1; ++i) {
      if(Mem::load(blocks[i]) != ONES) {
        return false;
      }
    }
    return Mem::load(blocks[i]) == ONES_LAST;
  }

  CUDA constexpr bool any() const {
    for(int i = 0; i < BLOCKS; ++i) {
      if(Mem::load(blocks[i]) > ZERO) {
        return true;
      }
    }
    return false;
  }

  CUDA constexpr bool none() const {
    for(int i = 0; i < BLOCKS; ++i) {
      if(Mem::load(blocks[i]) != ZERO) {
        return false;
      }
    }
    return true;
  }

  CUDA constexpr size_t size() const {
    return MAX_SIZE;
  }

  CUDA constexpr size_t count() const {
    size_t bits_at_one = 0;
    for(int i = 0; i < BLOCKS; ++i){
      bits_at_one += popcount(blocks[i]);
    }
    return bits_at_one;
  }

  CUDA constexpr bitset& set() {
    int i = 0;
    for(; i < BLOCKS - 1; ++i) {
      store(blocks[i], ONES);
    }
    store(blocks[i], ONES_LAST);
    return *this;
  }

  CUDA constexpr bitset& set(size_t pos) {
    assert(pos < MAX_SIZE);
    store_block(pos, load_block(pos) | bit_of(pos));
    return *this;
  }

  CUDA constexpr bitset& set(size_t pos, bool value) {
    assert(pos < MAX_SIZE);
    return value ? set(pos) : reset(pos);
  }

  CUDA constexpr bitset& reset() {
    for(int i = 0; i < BLOCKS; ++i) {
      store(blocks[i], ZERO);
    }
    return *this;
  }

  CUDA constexpr bitset& reset(size_t pos) {
    assert(pos < MAX_SIZE);
    store_block(pos, load_block(pos) & ~bit_of(pos));
    return *this;
  }

  CUDA constexpr bitset& flip() {
    int i = 0;
    for(; i < BLOCKS - 1; ++i) {
      store(blocks[i], ~Mem::load(blocks[i]));
    }
    store(blocks[i], ONES_LAST & ~Mem::load(blocks[i]));
    return *this;
  }

  CUDA constexpr bitset& flip(size_t pos) {
    assert(pos < MAX_SIZE);
    store_block(pos, load_block(pos) ^ bit_of(pos));
    return *this;
  }

  template <class Mem2>
  CUDA constexpr bool is_subset_of(const bitset<N, Mem2, T>& other) const {
    for(int i = 0; i < BLOCKS; ++i) {
      T block = Mem::load(blocks[i]);
      if((block & Mem2::load(other.blocks[i])) != block) {
        return false;
      }
    }
    return true;
  }

  template <class Mem2>
  CUDA constexpr bool is_proper_subset_of(const bitset<N, Mem2, T>& other) const {
    bool proper = false;
    for(int i = 0; i < BLOCKS; ++i) {
      T block = Mem::load(blocks[i]);
      T block2 = Mem2::load(other.blocks[i]);
      if((block & block2) != block) {
        return false;
      }
      proper |= block != block2;
    }
    return proper;
  }

  template<class Mem2>
  CUDA constexpr bitset& operator&=(const bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      store(blocks[i], Mem::load(blocks[i]) & Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  template<class Mem2>
  CUDA constexpr bitset& operator|=(const bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      store(blocks[i], Mem::load(blocks[i]) | Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  template<class Mem2>
  CUDA constexpr bitset& operator^=(const bitset<N, Mem2, T>& other) {
    for(int i = 0; i < BLOCKS; ++i) {
      store(blocks[i], Mem::load(blocks[i]) ^ Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  CUDA constexpr bitset operator~() const {
    return bitset(*this).flip();
  }

  template<class Mem2>
  CUDA constexpr bool operator==(const bitset<N, Mem2, T>& other) const {
    for(int i = 0; i < BLOCKS; ++i) {
      if(blocks[i] != other.blocks[i]) {
        return false;
      }
    }
    return true;
  }

  template<class Mem2>
  CUDA constexpr bool operator!=(const bitset<N, Mem2, T>& other) const {
    return !(*this == other);
  }

  CUDA constexpr int countl_zero() const {
    int k = BLOCKS - 1;
    if(blocks[k] > ZERO) {
      return ::battery::countl_zero(blocks[k]) - PADDING_LAST_BLOCK;
    }
    --k;
    int i = 0;
    for(; k >= 0 && blocks[k] == ZERO; --k, ++i) {}
    return i * BITS_PER_BLOCK
         + BITS_LAST_BLOCK
         + (k == -1 ? 0 : ::battery::countl_zero(blocks[k]));
  }

  CUDA constexpr int countl_one() const {
    int k = BLOCKS - 1;
    if(blocks[k] != ONES_LAST) {
      return battery::countl_one((T)(blocks[k] << PADDING_LAST_BLOCK));
    }
    --k;
    int i = 0;
    for(; k >= 0 && blocks[k] == ONES; --k, ++i) {}
    return i * BITS_PER_BLOCK
         + BITS_LAST_BLOCK
         + (k == -1 ? 0 : ::battery::countl_one(blocks[k]));
  }

  CUDA constexpr int countr_zero() const {
    int k = 0;
    for(; k < BLOCKS && blocks[k] == ZERO; ++k) {}
    return k * BITS_PER_BLOCK
        + (k == BLOCKS ? -PADDING_LAST_BLOCK : ::battery::countr_zero(blocks[k]));
  }

  CUDA constexpr int countr_one() const {
    int k = 0;
    for(; k < BLOCKS && blocks[k] == ONES; ++k) {}
    return k * BITS_PER_BLOCK
        + (k == BLOCKS ? 0 : ::battery::countr_one(blocks[k]));
  }

  CUDA constexpr void print() const {
    for(int i = size() - 1; i >= 0; --i) {
      printf("%c", test(i) ? '1' : '0');
    }
  }

  template <class Allocator>
  CUDA string<Allocator> to_string(Allocator allocator = Allocator()) const {
    string<Allocator> bits_str(size(), allocator);
    for(int i = size() - 1, j = 0; i >= 0; --i, ++j) {
      bits_str[j] = test(i) ? '1' : '0';
    }
    return bits_str;
  }
};

template<size_t N, class Mem, class T>
CUDA constexpr bitset<N, Mem, T> operator&(const bitset<N, Mem, T>& lhs, const bitset<N, Mem, T>& rhs) {
  return bitset<N, Mem, T>(lhs) &= rhs;
}

template<size_t N, class Mem, class T>
CUDA constexpr bitset<N, Mem, T> operator|(const bitset<N, Mem, T>& lhs, const bitset<N, Mem, T>& rhs) {
  return bitset<N, Mem, T>(lhs) |= rhs;
}

template<size_t N, class Mem, class T>
CUDA constexpr bitset<N, Mem, T> operator^(const bitset<N, Mem, T>& lhs, const bitset<N, Mem, T>& rhs) {
  return bitset<N, Mem, T>(lhs) ^= rhs;
}

} // namespace battery

#endif
