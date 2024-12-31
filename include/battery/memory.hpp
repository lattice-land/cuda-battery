// Copyright 2022 Pierre Talbot

#ifndef CUDA_BATTERY_MEMORY_HPP
#define CUDA_BATTERY_MEMORY_HPP

/** \file memory.hpp
 *  An abstraction of memory load and store useful to write a single version of an algorithm working either sequentially or in parallel, and on CPU or GPU.
 * Note that these classes are mainly designed to work with a relaxed memory ordering; we are unsure of their applicability to other kinds of memory ordering.
 */

#include <cassert>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
  #include <cuda/atomic>
#else
  #include <atomic>
#endif

#include "utility.hpp"

namespace battery {

template <class A>
class copyable_atomic: public A {
public:
  copyable_atomic() = default;
  CUDA copyable_atomic(typename A::value_type x): A(x) {}
  copyable_atomic(const copyable_atomic& other): A(other.load()) {}
  copyable_atomic(copyable_atomic&& other): A(other.load()) {}
  copyable_atomic& operator=(const copyable_atomic& other) {
    this->store(other.load());
    return *this;
  }
  copyable_atomic& operator=(copyable_atomic&& other) {
    this->store(other.load());
    return *this;
  }
};

/** Represent the memory of a variable that cannot be accessed by multiple threads. */
template <bool read_only>
class memory {
public:
  template <class T> using atomic_type = T;

  /** Indicate this memory is written by a single thread. */
  constexpr static const bool sequential = true;

public:
  template <class T>
  CUDA INLINE static constexpr T load(const atomic_type<T>& a) {
    return a;
  }

  template <class T>
  CUDA INLINE static constexpr std::enable_if_t<!read_only, void> store(atomic_type<T>& a, T v) {
    a = v;
  }

  template <class T>
  CUDA INLINE static constexpr std::enable_if_t<!read_only, T> exchange(atomic_type<T>& a, T v) {
    return std::exchange(a, std::move(v));
  }
};

using local_memory = memory<false>;
using read_only_memory = memory<true>;

#ifdef __CUDACC__

/** Memory load and store operations relative to a cuda scope (per-thread, block, grid, ...) and given a certain memory order (by default relaxed). */
template <cuda::thread_scope scope, cuda::memory_order mem_order = cuda::memory_order_relaxed>
class atomic_memory_scoped {
public:
  template <class T> using atomic_type = copyable_atomic<cuda::atomic<T, scope>>;
  constexpr static const bool sequential = false;

  template <class T>
  CUDA INLINE static T load(const atomic_type<T>& a) {
    return a.load(mem_order);
  }

  template <class T>
  CUDA INLINE static void store(atomic_type<T>& a, T v) {
    a.store(v, mem_order);
  }

  template <class T>
  CUDA INLINE static T exchange(atomic_type<T>& a, T v) {
    return a.exchange(v, mem_order);
  }
};

using atomic_memory_block = atomic_memory_scoped<cuda::thread_scope_block>;
using atomic_memory_grid = atomic_memory_scoped<cuda::thread_scope_device>;
using atomic_memory_multi_grid = atomic_memory_scoped<cuda::thread_scope_system>;

#endif // __CUDACC__

#ifdef __CUDACC__
  /// @private
  namespace impl {
    template <class T>
    using atomic_t = cuda::std::atomic<T>;
  }
  /// @private
  using memory_order = cuda::std::memory_order;
  /// @private
  constexpr memory_order memory_order_relaxed = cuda::std::memory_order_relaxed;
  /// @private
  constexpr memory_order memory_order_seq_cst = cuda::std::memory_order_seq_cst;
#else
  /// @private
  namespace impl {
    template <class T>
    using atomic_t = std::atomic<T>;
  }
  /// @private
  using memory_order = std::memory_order;
  /// @private
  constexpr memory_order memory_order_relaxed = std::memory_order_relaxed;
  /// @private
  constexpr memory_order memory_order_seq_cst = std::memory_order_seq_cst;
#endif

/** Use the standard C++ atomic type, either provided by libcudacxx if we compile with a CUDA compiler, or through the STL otherwise. */
template <memory_order mem_order = memory_order_relaxed>
class atomic_memory {
public:
  template <class T> using atomic_type = copyable_atomic<impl::atomic_t<T>>;
  constexpr static const bool sequential = false;

  template <class T>
  CUDA INLINE static T load(const atomic_type<T>& a) {
    return a.load(mem_order);
  }

  template <class T>
  CUDA INLINE static void store(atomic_type<T>& a, T v) {
    a.store(v, mem_order);
  }

  template <class T>
  CUDA INLINE static T exchange(atomic_type<T>& a, T v) {
    return a.exchange(v, mem_order);
  }
};

}

#endif
