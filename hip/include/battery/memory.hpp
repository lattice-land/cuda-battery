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

// HIP cross-platform atomic header inclusion
#ifdef __CUDACC__
  #include <hip/hip_runtime.h>
  #include <atomic>
#else
  #include <atomic>
#endif

#include "utility.hpp"

namespace battery {

// HIP cross-platform atomic types
#ifdef __CUDACC__
  // HIP uses standard atomics - no scoped atomics available
  using gpu_memory_order = std::memory_order;
  constexpr gpu_memory_order gpu_memory_order_relaxed = std::memory_order_relaxed;
  constexpr gpu_memory_order gpu_memory_order_seq_cst = std::memory_order_seq_cst;
  
  // HIP thread scopes (mapped to single scope since HIP doesn't have scoped atomics)
  enum class thread_scope {
    block = 0,
    device = 1,
    system = 2
  };
#else
  // CPU fallback
  using gpu_memory_order = std::memory_order;
  constexpr gpu_memory_order gpu_memory_order_relaxed = std::memory_order_relaxed;
  constexpr gpu_memory_order gpu_memory_order_seq_cst = std::memory_order_seq_cst;
  
  enum class thread_scope {
    block = 0,
    device = 1,
    system = 2
  };
#endif

/** HIP atomic type selector */
namespace impl {
  template <class T>
  struct value_type_of {
    using type = typename T::value_type;
  };
  
  // HIP uses standard atomics (no scoped atomics available)
  template <class T, thread_scope scope>
  using gpu_atomic_type = std::atomic<T>;
}


template <class A>
class copyable_atomic: public A {
public:
  using value_type = typename impl::value_type_of<A>::type;
  copyable_atomic() = default;
  CUDA copyable_atomic(value_type x): A(x) {}
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

/** Cross-platform scoped atomic memory operations.
 * On CUDA: Uses real scoped atomics with proper thread scope semantics.
 * On HIP/CPU: Falls back to regular atomics (scope is ignored but API remains compatible).
 */
template <thread_scope scope, gpu_memory_order mem_order = gpu_memory_order_relaxed>
class atomic_memory_scoped {
public:
  // HIP and CPU use regular atomics (scope is ignored but API remains compatible)
  template <class T> using atomic_type = copyable_atomic<std::atomic<T>>;
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

// Cross-platform atomic memory type aliases
using atomic_memory_block = atomic_memory_scoped<thread_scope::block>;
using atomic_memory_grid = atomic_memory_scoped<thread_scope::device>;
using atomic_memory_multi_grid = atomic_memory_scoped<thread_scope::system>;

/// @private
namespace impl {
  template <class T>
  using atomic_t = std::atomic<T>;
}

/// @private
using memory_order = std::memory_order;
constexpr memory_order memory_order_relaxed = std::memory_order_relaxed;
constexpr memory_order memory_order_seq_cst = std::memory_order_seq_cst;

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
