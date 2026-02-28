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
#include "utility.hpp"

#ifdef BATTERY_CUDA_BACKEND
  #include <cuda/atomic>
#elif defined(BATTERY_HIP_BACKEND)
  #include <hip/hip_runtime.h>
  #include <atomic>
#else
  #include <atomic>
#endif

namespace battery {

/** This is to be deleted in the future, just there because atomic in CUDA 12.4 does not support ::value_type. */
namespace impl {
  template <class T>
  struct value_type_of {
    using type = typename T::value_type;
  };
#ifdef BATTERY_CUDA_BACKEND
  template <class V, cuda::thread_scope Scope>
  struct value_type_of<cuda::atomic<V, Scope>> {
    using type = V;
  };
#endif

#ifdef BATTERY_HIP_BACKEND
  /** HIP scope constants mapping to __HIP_MEMORY_SCOPE_* values. */
  constexpr int hip_scope_block      = __HIP_MEMORY_SCOPE_WORKGROUP;
  constexpr int hip_scope_device     = __HIP_MEMORY_SCOPE_AGENT;
  constexpr int hip_scope_system     = __HIP_MEMORY_SCOPE_SYSTEM;

  /**
   * Thin wrapper around __hip_atomic_* intrinsics that mirrors the
   * cuda::atomic<V, Scope> interface expected by copyable_atomic.
   * Works in both host and device code under hipcc.
   */
  template <class T, int Scope>
  class hip_atomic_wrapper {
    T value_;
  public:
    using value_type = T;
    CUDA hip_atomic_wrapper() = default;
    CUDA hip_atomic_wrapper(T v) : value_(v) {}

    CUDA T load(int order = __ATOMIC_RELAXED) const {
      return __hip_atomic_load(&value_, order, Scope);
    }
    CUDA void store(T v, int order = __ATOMIC_RELAXED) {
      __hip_atomic_store(&value_, v, order, Scope);
    }
    CUDA T exchange(T v, int order = __ATOMIC_RELAXED) {
      return __hip_atomic_exchange(&value_, v, order, Scope);
    }
  };
#endif // BATTERY_HIP_BACKEND
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

#ifdef BATTERY_CUDA_BACKEND

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

#endif // BATTERY_CUDA_BACKEND

#ifdef BATTERY_HIP_BACKEND

/** Memory load and store operations with an explicit HIP memory scope and
 *  a given memory order (relaxed by default).
 *  The Scope template parameter uses the __HIP_MEMORY_SCOPE_* integer constants,
 *  exposed through battery::impl::hip_scope_{block,device,system}.
 */
template <int Scope, int mem_order = __ATOMIC_RELAXED>
class atomic_memory_scoped {
public:
  template <class T> using atomic_type = copyable_atomic<impl::hip_atomic_wrapper<T, Scope>>;
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

using atomic_memory_block      = atomic_memory_scoped<impl::hip_scope_block>;
using atomic_memory_grid       = atomic_memory_scoped<impl::hip_scope_device>;
using atomic_memory_multi_grid = atomic_memory_scoped<impl::hip_scope_system>;

#endif // BATTERY_HIP_BACKEND

#ifdef BATTERY_CUDA_BACKEND
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
