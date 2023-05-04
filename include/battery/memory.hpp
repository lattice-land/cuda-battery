// Copyright 2022 Pierre Talbot

#ifndef MEMORY_HPP
#define MEMORY_HPP

/** An abstraction of memory useful to write a single version of an algorithm working either sequentially or in parallel, and on CPU or GPU. */

#include <cassert>
#include <type_traits>
#include "allocator.hpp"

#ifdef __NVCC__
  #include <cuda/atomic>
#else
  #include <atomic>
#endif

namespace battery {

/** Represent the memory of a variable that cannot be accessed by multiple threads (e.g., allocated on the stack), thus not needing any allocator.
    In other terms, a variable stored locally to a thread. */
class local_memory {
public:
  template <class T> using atomic_type = T;

  /** Indicate this memory is written by a single thread. */
  constexpr static const bool sequential = true;

  template <class T>
  CUDA static constexpr T load(const atomic_type<T>& a) {
    return a;
  }

  template <class T>
  CUDA static constexpr void store(atomic_type<T>& a, T v) {
    a = v;
  }
};

template <class Allocator, bool read_only = false>
class memory {
public:
  using allocator_type = Allocator;
  template <class T> using atomic_type = T;
  constexpr static const bool sequential = true;

private:
  allocator_type alloc;

public:
  CUDA constexpr memory(allocator_type alloc): alloc(alloc) {}
  CUDA constexpr memory(const memory& seq): alloc(seq.alloc) {}

  template <class T>
  CUDA static constexpr T load(const atomic_type<T>& a) {
    return a;
  }

  template <class T>
  CUDA static constexpr std::enable_if_t<!read_only, void> store(atomic_type<T>& a, T v) {
    a = v;
  }

  CUDA constexpr allocator_type get_allocator() const {
    return alloc;
  }
};

template <class Allocator>
using read_only_memory = memory<Allocator, true>;

#ifdef __NVCC__

template <class Allocator, cuda::thread_scope scope, cuda::memory_order mem_order = cuda::memory_order_relaxed>
class atomic_memory_scoped {
public:
  using allocator_type = Allocator;
  template <class T> using atomic_type = cuda::atomic<T, scope>;
  constexpr static const bool sequential = false;

private:
  allocator_type alloc;

public:
  CUDA constexpr atomic_memory_scoped(allocator_type alloc): alloc(alloc) {}
  CUDA constexpr atomic_memory_scoped(const atomic_memory_scoped& seq): alloc(seq.alloc) {}

  template <class T>
  CUDA static T load(const atomic_type<T>& a) {
    return a.load(mem_order);
  }

  template <class T>
  CUDA static void store(atomic_type<T>& a, T v) {
    a.store(v, mem_order);
  }

  CUDA constexpr allocator_type get_allocator() const {
    return alloc;
  }
};

template <class Allocator, cuda::memory_order mem_order = cuda::memory_order_relaxed>
using atomic_memory_block = atomic_memory_scoped<Allocator, cuda::thread_scope_block, mem_order>;

template <class Allocator, cuda::memory_order mem_order = cuda::memory_order_relaxed>
using atomic_memory_grid = atomic_memory_scoped<Allocator, cuda::thread_scope_device, mem_order>;

template <class Allocator, cuda::memory_order mem_order = cuda::memory_order_relaxed>
using atomic_memory_multi_grid = atomic_memory_scoped<Allocator, cuda::thread_scope_system, mem_order>;

#endif // __NVCC__

#ifdef __NVCC__
  namespace impl {
    template <class T>
    using atomic_t = cuda::std::atomic<T>;
  }
  using memory_order = cuda::std::memory_order;
  constexpr memory_order memory_order_relaxed = cuda::std::memory_order_relaxed;
  constexpr memory_order memory_order_seq_cst = cuda::std::memory_order_seq_cst;
#else
  namespace impl {
    template <class T>
    using atomic_t = std::atomic<T>;
  }
  using memory_order = std::memory_order;
  constexpr memory_order memory_order_relaxed = std::memory_order_relaxed;
  constexpr memory_order memory_order_seq_cst = std::memory_order_seq_cst;
#endif

template <class Allocator, memory_order mem_order = memory_order_relaxed>
class atomic_memory {
public:
  using allocator_type = Allocator;
  template <class T> using atomic_type = impl::atomic_t<T>;
  constexpr static const bool sequential = false;

private:
  allocator_type alloc;

public:
  CUDA constexpr atomic_memory(allocator_type alloc): alloc(alloc) {}
  CUDA constexpr atomic_memory(const atomic_memory& am): alloc(am.alloc) {}

  template <class T>
  CUDA static T load(const atomic_type<T>& a) {
    return a.load(mem_order);
  }

  template <class T>
  CUDA static void store(atomic_type<T>& a, T v) {
    a.store(v, mem_order);
  }

  CUDA constexpr allocator_type get_allocator() const {
    return alloc;
  }
};

}

#endif // MEMORY_HPP
