// Copyright 2023 Pierre Talbot

#ifndef CUDA_BATTERY_DYNAMIC_BITSET_HPP
#define CUDA_BATTERY_DYNAMIC_BITSET_HPP

#include <cstdio>
#include <cassert>
#include "utility.hpp"
#include "string.hpp"
#include "vector.hpp"

/** \file dynamic_bitset.hpp
 * This class is similar to `bitset` but the size is specified at runtime in the constructor.
 * For simplicity of implementation, the size is a multiple of sizeof(T); that could be improved in future version if needed.
 */

namespace battery {

/**
 * \tparam `Mem` is a memory consistency, for instance `local_memory`, see `memory.hpp`.
 * \tparam `T` is the underlying type defining the blocks of the bitset.
*/
template <class Mem, class Allocator = standard_allocator, class T = unsigned long long>
class dynamic_bitset {
public:
  using memory_type = Mem;
  using allocator_type = Allocator;
private:
  constexpr static const size_t BITS_PER_BLOCK = sizeof(T) * CHAR_BIT;

  // Would be better to have constexpr, but it gives an error "undefined in device code", probably because of the call to the constructor of `T`.
  #define ZERO (T{0})
  #define ONES (~T{0})

  using block_type = typename Mem::template atomic_type<T>;
  using this_type = dynamic_bitset<Mem, Allocator, T>;

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
  vector<block_type, allocator_type> blocks;

private:
  CUDA constexpr size_t num_blocks(size_t n) {
    return n / BITS_PER_BLOCK + (n % BITS_PER_BLOCK != 0);
  }

public:
  template <class Mem2, class Allocator2, class T2>
  friend class dynamic_bitset;

  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "Block of a bitset must be defined on an unsigned integer type.");

  CUDA dynamic_bitset(const allocator_type& alloc = allocator_type())
   : blocks(alloc) {}

  /** Create a bitset with a size of at least `num_bits`. */
  CUDA dynamic_bitset(size_t at_least_num_bits, const allocator_type& alloc = allocator_type())
    : blocks(num_blocks(at_least_num_bits), ZERO, alloc) {}

  CUDA dynamic_bitset(const char* bit_str, const allocator_type& alloc = allocator_type())
   : blocks(num_blocks(strlen(bit_str)), ZERO, alloc)
  {
    for(size_t i = strlen(bit_str), j = 0; i > 0; --i, ++j) {
      if(bit_str[i-1] == '1') {
        set(j);
      }
    }
  }

  /** Set all bits in the range [start..end] to `1` and create a bitset with at least `end+1` bits.
   * `end >= start`. */
  CUDA dynamic_bitset(size_t start, size_t end, const allocator_type& alloc = allocator_type())
   : dynamic_bitset(end+1, alloc) {
    assert(end >= start);
    int block_start = start / BITS_PER_BLOCK;
    int block_end = end / BITS_PER_BLOCK;
    store(blocks[block_start], ONES << (start % BITS_PER_BLOCK));
    for(int k = block_start + 1; k <= block_end; ++k) {
      store(blocks[k], ONES);
    }
    store(blocks[block_end], Mem::load(blocks[block_end]) & (ONES >> ((BITS_PER_BLOCK-(end % BITS_PER_BLOCK)-1))));
  }

  template<class Mem2, class Alloc2>
  CUDA dynamic_bitset(const dynamic_bitset<Mem2, Alloc2, T>& other,
    const allocator_type& alloc = allocator_type())
   : blocks(other.blocks.size(), ZERO, alloc)
  {
    for(int i = 0; i < blocks.size(); ++i) {
      Mem::store(blocks[i], Mem2::load(other.blocks[i]));
    }
  }

  CUDA dynamic_bitset(const this_type& from)
   : this_type(from, from.get_allocator()) {}

  dynamic_bitset(this_type&&) = default;

  CUDA this_type& operator=(this_type&& other) {
    blocks = std::move(other.blocks);
    return *this;
  }

  CUDA allocator_type get_allocator() const {
    return blocks.get_allocator();
  }

private:
  CUDA INLINE constexpr void store(block_type& block, T data) {
    Mem::store(block, data);
  }

  CUDA INLINE constexpr block_type& block_of(size_t pos) {
    return blocks[pos / BITS_PER_BLOCK];
  }

  CUDA INLINE constexpr const block_type& block_of(size_t pos) const {
    return blocks[pos / BITS_PER_BLOCK];
  }

  CUDA INLINE constexpr T load_block(size_t pos) const {
    return Mem::load(block_of(pos));
  }

  CUDA INLINE constexpr void store_block(size_t pos, T data) {
    store(block_of(pos), data);
  }

  CUDA INLINE constexpr T bit_of(size_t pos) const {
    return static_cast<T>(1) << (pos % BITS_PER_BLOCK);
  }

public:
  // This is not thread-safe.
  CUDA constexpr void swap(this_type& other) {
    if(size() != other.size()) {
      battery::swap(*this, other);
      return;
    }
    for(int i = 0; i < blocks.size(); ++i) {
      Mem::store(blocks[i], Mem::load(blocks[i]) ^ Mem::load(other.blocks[i]));
      Mem::store(other.blocks[i], Mem::load(blocks[i]) ^ Mem::load(other.blocks[i]));
      Mem::store(blocks[i], Mem::load(blocks[i]) ^ Mem::load(other.blocks[i]));
    }
  }

  CUDA constexpr bool test(size_t pos) const {
    assert(pos < size());
    return load_block(pos) & bit_of(pos);
  }

  CUDA constexpr bool all() const {
    for(int i = 0; i < blocks.size(); ++i) {
      if(Mem::load(blocks[i]) != ONES) {
        return false;
      }
    }
    return true;
  }

  CUDA constexpr bool any() const {
    for(int i = 0; i < blocks.size(); ++i) {
      if(Mem::load(blocks[i]) != ZERO) {
        return true;
      }
    }
    return false;
  }

  CUDA constexpr bool none() const {
    for(int i = 0; i < blocks.size(); ++i) {
      if(Mem::load(blocks[i]) != ZERO) {
        return false;
      }
    }
    return true;
  }

  CUDA constexpr size_t size() const {
    return BITS_PER_BLOCK * blocks.size();
  }

  // Only the values of the first `at_least_num_bits` are copied in the new resized bitset.
  CUDA void resize(size_t at_least_num_bits) {
    if(num_blocks(at_least_num_bits) == blocks.size()) {
      return;
    }
    // NOTE: We cannot call vector.resize because it does not support resizing non-copyable, non-movable types such as atomics.
    // Therefore, we implement our own resizing function using explicit load and store operations.
    this_type bitset2(at_least_num_bits, get_allocator());
    for(int i = 0; i < at_least_num_bits && i < size(); ++i) {
      bitset2.set(i, test(i));
    }
    blocks.swap(bitset2.blocks);
  }

  CUDA constexpr size_t count() const {
    size_t bits_at_one = 0;
    for(int i = 0; i < blocks.size(); ++i){
      bits_at_one += popcount(Mem::load(blocks[i]));
    }
    return bits_at_one;
  }

  CUDA constexpr dynamic_bitset& set() {
    for(int i = 0; i < blocks.size(); ++i) {
      store(blocks[i], ONES);
    }
    return *this;
  }

  CUDA constexpr dynamic_bitset& set(size_t pos) {
    assert(pos < size());
    store_block(pos, load_block(pos) | bit_of(pos));
    return *this;
  }

  CUDA constexpr dynamic_bitset& set(size_t pos, bool value) {
    assert(pos < size());
    return value ? set(pos) : reset(pos);
  }

  CUDA constexpr dynamic_bitset& reset() {
    for(int i = 0; i < blocks.size(); ++i) {
      store(blocks[i], ZERO);
    }
    return *this;
  }

  CUDA constexpr dynamic_bitset& reset(size_t pos) {
    assert(pos < size());
    store_block(pos, load_block(pos) & ~bit_of(pos));
    return *this;
  }

  CUDA constexpr dynamic_bitset& flip() {
    for(int i = 0; i < blocks.size(); ++i) {
      store(blocks[i], ~Mem::load(blocks[i]));
    }
    return *this;
  }

  CUDA constexpr dynamic_bitset& flip(size_t pos) {
    assert(pos < size());
    store_block(pos, load_block(pos) ^ bit_of(pos));
    return *this;
  }

  template <class Mem2, class Alloc2>
  CUDA constexpr bool is_subset_of(const dynamic_bitset<Mem2, Alloc2, T>& other) const {
    size_t min_size = min(blocks.size(), other.blocks.size());
    for(int i = 0; i < min_size; ++i) {
      T block = Mem::load(blocks[i]);
      if((block & Mem2::load(other.blocks[i])) != block) {
        return false;
      }
    }
    if(size() > other.size()) {
      for(int i = min_size; i < blocks.size(); ++i) {
        if(Mem::load(blocks[i]) != ZERO) {
          return false;
        }
      }
    }
    return true;
  }

  template <class Mem2, class Alloc2>
  CUDA constexpr bool is_proper_subset_of(const dynamic_bitset<Mem2, Alloc2, T>& other) const {
    bool proper = false;
    size_t min_size = min(blocks.size(), other.blocks.size());
    for(int i = 0; i < min_size; ++i) {
      T block = Mem::load(blocks[i]);
      T block2 = Mem2::load(other.blocks[i]);
      if((block & block2) != block) {
        return false;
      }
      proper |= block != block2;
    }
    if(proper && blocks.size() > other.blocks.size()) {
      for(int i = min_size; i < blocks.size(); ++i) {
        if(Mem::load(blocks[i]) != ZERO) {
          return false;
        }
      }
    }
    else if(!proper && blocks.size() < other.blocks.size()) {
      for(int i = min_size; i < other.blocks.size(); ++i) {
        if(Mem2::load(other.blocks[i]) != ZERO) {
          return true;
        }
      }
    }
    return proper;
  }

  template<class Mem2, class Alloc2>
  CUDA constexpr dynamic_bitset& operator&=(const dynamic_bitset<Mem2, Alloc2, T>& other) {
    for(int i = 0; i < blocks.size(); ++i) {
      store(blocks[i], Mem::load(blocks[i]) & Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  template<class Mem2, class Alloc2>
  CUDA constexpr dynamic_bitset& operator|=(const dynamic_bitset<Mem2, Alloc2, T>& other) {
    for(int i = 0; i < blocks.size(); ++i) {
      store(blocks[i], Mem::load(blocks[i]) | Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  template<class Mem2, class Alloc2>
  CUDA constexpr dynamic_bitset& operator^=(const dynamic_bitset<Mem2, Alloc2, T>& other) {
    for(int i = 0; i < blocks.size(); ++i) {
      store(blocks[i], Mem::load(blocks[i]) ^ Mem2::load(other.blocks[i]));
    }
    return *this;
  }

  CUDA constexpr dynamic_bitset operator~() const {
    return dynamic_bitset(*this).flip();
  }

  template<class Mem2, class Alloc2>
  CUDA constexpr bool operator==(const dynamic_bitset<Mem2, Alloc2, T>& other) const {
    for(int i = 0; i < blocks.size(); ++i) {
      if(Mem::load(blocks[i]) != Mem2::load(other.blocks[i])) {
        return false;
      }
    }
    return true;
  }

  template<class Mem2, class Alloc2>
  CUDA constexpr bool operator!=(const dynamic_bitset<Mem2, Alloc2, T>& other) const {
    return !(*this == other);
  }

  CUDA constexpr void print() const {
    for(int i = size() - 1; i >= 0; --i) {
      printf("%c", test(i) ? '1' : '0');
    }
  }

  template <class Alloc2>
  CUDA string<Alloc2> to_string(Alloc2 allocator = Alloc2()) const {
    string<Alloc2> bits_str(size(), allocator);
    for(int i = size() - 1, j = 0; i >= 0; --i, ++j) {
      bits_str[j] = test(i) ? '1' : '0';
    }
    return bits_str;
  }
};

template<class Mem, class Alloc, class T>
CUDA constexpr dynamic_bitset<Mem, Alloc, T> operator&(const dynamic_bitset<Mem, Alloc, T>& lhs, const dynamic_bitset<Mem, Alloc, T>& rhs) {
  return dynamic_bitset<Mem, Alloc, T>(lhs) &= rhs;
}

template<class Mem, class Alloc, class T>
CUDA constexpr dynamic_bitset<Mem, Alloc, T> operator|(const dynamic_bitset<Mem, Alloc, T>& lhs, const dynamic_bitset<Mem, Alloc, T>& rhs) {
  return dynamic_bitset<Mem, Alloc, T>(lhs) |= rhs;
}

template<class Mem, class Alloc, class T>
CUDA constexpr dynamic_bitset<Mem, Alloc, T> operator^(const dynamic_bitset<Mem, Alloc, T>& lhs, const dynamic_bitset<Mem, Alloc, T>& rhs) {
  return dynamic_bitset<Mem, Alloc, T>(lhs) ^= rhs;
}

#undef ZERO
#undef ONES

} // namespace battery

#endif
