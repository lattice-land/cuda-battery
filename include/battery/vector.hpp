// Copyright 2021 Pierre Talbot

#ifndef CUDA_BATTERY_VECTOR_HPP
#define CUDA_BATTERY_VECTOR_HPP

#include "utility.hpp"
#include "allocator.hpp"
#include <memory>
#include <initializer_list>
#include <vector>

/** \file vector.hpp
 * A partial implementation of `std::vector`, with additional constructors to convert from `std::vector`.
 * The allocator is scoped, meaning it will be passed to the constructor of `T` if `T` provides a suited constructor with an allocator.
*/

namespace battery {

template<class T, class Allocator = standard_allocator>
class vector {

public:
  using value_type = T;
  using allocator_type = Allocator;
  using this_type = vector<value_type, allocator_type>;

private:
  static constexpr const size_t GROWING_FACTOR = 3;
  size_t n;
  size_t cap;
  allocator_type allocator;
  value_type* data_;

  CUDA T* allocate() {
    if(cap > 0) {
      return static_cast<T*>(allocator.allocate(sizeof(T) * cap));
    }
    else {
      return nullptr;
    }
  }

  CUDA void deallocate() {
    allocator.deallocate(data_);
  }

  CUDA NI void reallocate(size_t new_cap) {
    size_t n2 = n;
    value_type* data2 = data_;
    cap = new_cap;
    n = min(new_cap, n); // in case we shrink the initial array.
    data_ = allocate();
    for(size_t i = 0; i < n; ++i) {
      if constexpr(std::is_move_constructible_v<value_type>) {
        new(&data_[i]) value_type(std::move(data2[i]));
      }
      else {
        new(&data_[i]) value_type(data2[i]);
      }
    }
    if constexpr(!std::is_trivially_destructible_v<value_type>) {
      // Free the old data array that has been reallocated.
      for(size_t i = 0; i < n2; ++i) {
        data2[i].~T();
      }
    }
    allocator.deallocate(data2);
  }

  // Reallocate memory if the current array is full.
  CUDA void reallocate() {
    if(n == cap) {
      reallocate(max(cap, size_t(1)) * GROWING_FACTOR);
    }
  }

  template<class T2, class Allocator2> friend class vector;

  CUDA void inplace_new(size_t i) {
    if constexpr(std::is_constructible<value_type, const allocator_type&>{}) {
      new(&data_[i]) value_type(allocator);
    }
    else {
      new(&data_[i]) value_type{};
    }
  }

  template <class U>
  CUDA void inplace_new(size_t i, const U& value) {
    if constexpr(std::is_constructible<value_type, const U&, const allocator_type&>{}) {
      new(&data_[i]) value_type(value, allocator);
    }
    else {
      new(&data_[i]) value_type(value);
    }
  }

public:
  /** Allocate an array of size `n` using `allocator`, with `n` default-inserted instances of `T`.
   *  `Allocator` is scoped, meaning it will be passed to the constructor of `T` if `T(const Allocator&)` exists, otherwise `T()` is called. */
  CUDA NI vector(size_t n, const allocator_type& alloc = allocator_type()):
    n(n), cap(n), allocator(alloc), data_(allocate())
  {
    for(size_t i = 0; i < n; ++i) {
      inplace_new(i);
    }
  }

  /** Default constructor.*/
  CUDA vector(const allocator_type& alloc = allocator_type{})
   : n(0), cap(0), allocator(alloc), data_(nullptr) {}

  /** Allocate an array of size `n` using `allocator`.
      Initialize the elements of the array with those of the array `from`.
      `Allocator` is scoped, meaning it will be passed to the constructor of `T` if `T(const T&, const Allocator&)` exists, otherwise `T(const T&)` is called.  */
  template <class U>
  CUDA NI vector(const U* from, size_t n, const allocator_type& alloc = allocator_type{})
   : n(n), cap(n), allocator(alloc), data_(allocate())
  {
    for(size_t i = 0; i < n; ++i) {
      inplace_new(i, from[i]);
    }
  }

  /** Copy constructor with an allocator, useful when the vector we copied from was declared using another allocator. */
  template <class U, class Allocator2>
  CUDA vector(const vector<U, Allocator2>& from, const allocator_type& alloc = allocator_type{})
   : this_type(from.data(), from.size(), alloc) {}

  /** Copy constructor with different underlying types. */
  template <class U>
  CUDA vector(const vector<U, allocator_type>& from)
   : this_type(from, from.allocator) {}

  /** Redefine the copy constructor to be sure it calls a constructor with an allocator. */
  CUDA vector(const this_type& from)
   : this_type(from, from.allocator) {}

  /** Initialize of an array of size `n` with each element initialized to `from` using `allocator`.
   * `Allocator` is scoped, meaning it will be passed to the constructor of `T` if `T(const T&, const Allocator&)` exists, otherwise `T(const T&)` is called.  */
  template <class U>
  CUDA NI vector(size_t n, const U& from, const allocator_type& alloc = allocator_type{})
   : n(n), cap(n), allocator(alloc), data_(allocate())
  {
    for(size_t i = 0; i < n; ++i) {
      inplace_new(i, from);
    }
  }

  CUDA NI vector(this_type&& other): this_type(other.allocator) {
    swap(other);
  }

  CUDA NI vector(std::initializer_list<T> init, const allocator_type& alloc = allocator_type{})
   : n(init.size()), cap(init.size()), allocator(alloc), data_(allocate())
  {
    size_t i = 0;
    for(const T& v : init) {
      inplace_new(i, v);
      ++i;
    }
  }

  template <class U, class Alloc2>
  vector(const std::vector<U, Alloc2>& other, const allocator_type& alloc = allocator_type{})
   : n(other.size()), cap(other.size()), allocator(alloc), data_(allocate())
  {
    for(size_t i = 0; i < n; ++i) {
      inplace_new(i, other[i]);
    }
  }

  CUDA NI ~vector() {
    if constexpr(!std::is_trivially_destructible_v<value_type>) {
      for(size_t i = 0; i < n; ++i) {
        data_[i].~T();
      }
    }
    allocator.deallocate(data_);
    data_ = nullptr;
    n = 0;
    cap = 0;
  }

  CUDA this_type& operator=(this_type&& other) {
    swap(other);
    return *this;
  }

private:
  template <class U, class Allocator2>
  CUDA this_type& assignment(const vector<U, Allocator2>& other) {
    reserve(other.size());
    constexpr bool fast_copy =
      #ifdef __CUDA_ARCH__
        std::is_same_v<value_type, U>
        && std::is_same_v<Allocator2, allocator_type>
        && std::is_trivially_copyable_v<value_type>;
      #else
        std::is_same_v<value_type, U>
        && std::is_trivially_copyable_v<value_type>;
      #endif

    if constexpr(fast_copy) {
      #ifdef __CUDA_ARCH__
        cudaMemcpyAsync(data_, other.data_, other.n * sizeof(value_type), cudaMemcpyDeviceToDevice);
      #else
        memcpy(data_, other.data_, other.n * sizeof(value_type));
      #endif
      if constexpr(!std::is_trivially_destructible_v<value_type>) {
        for(size_t i = other.n; i < n; ++i) {
          data_[i].~value_type();
        }
      }
      n = other.n;
      return *this;
    }
    else {
      for(size_t i = 0; i < other.n; ++i) {
        if constexpr(!std::is_trivially_destructible_v<value_type>) {
          if(i < n) {
            data_[i].~value_type();
          }
        }
        inplace_new(i, other.data_[i]);
      }
      if constexpr(!std::is_trivially_destructible_v<value_type>) {
        for(size_t i = other.n; i < n; ++i) {
          data_[i].~value_type();
        }
      }
      n = other.n;
      return *this;
    }
  }

public:
  CUDA NI this_type& operator=(const this_type& other) {
    return assignment(other);
  }

  /** Beware that this operator does not free the memory of `this`, the capacity remains unchanged. */
  template <class U, class Allocator2>
  CUDA NI this_type& operator=(const vector<U, Allocator2>& other) {
    return assignment(other);
  }

  CUDA allocator_type get_allocator() const {
    return allocator;
  }

  CUDA value_type& operator[](size_t i) {
    assert(i < n);
    return data_[i];
  }

  CUDA const value_type& operator[](size_t i) const {
    assert(i < n);
    return data_[i];
  }

  CUDA value_type& front() { assert(n > 0); return data_[0]; }
  CUDA const value_type& front() const { assert(n > 0); return data_[0]; }
  CUDA value_type& back() { assert(n > 0); return data_[n-1]; }
  CUDA const value_type& back() const { assert(n > 0); return data_[n-1]; }
  CUDA value_type* data() { return data_; }
  CUDA const value_type* data() const { return data_; }
  CUDA bool empty() const { return n == 0; }
  CUDA size_t size() const { return n; }
  CUDA void reserve(size_t new_cap) { if(new_cap > cap) reallocate(new_cap); }
  CUDA size_t capacity() const { return cap; }
  CUDA void shrink_to_fit() { if(cap > n) reallocate(n); }

  CUDA NI void clear() {
    size_t n2 = n;
    n = 0;
    for(size_t i = 0; i < n2; ++i) {
      data_[i].~T();
    }
  }

  CUDA void push_back(const T& value) {
    reallocate();
    inplace_new(n, value);
    ++n;
  }

  CUDA void push_back(T&& value) {
    reallocate();
    new(&data_[n]) T(std::forward<T>(value));
    ++n;
  }

  template<class... Args>
  CUDA value_type& emplace_back(Args&&... args) {
    reallocate();
    if constexpr(std::is_constructible<value_type, Args&&..., const allocator_type&>{}) {
      new(&data_[n]) value_type(std::forward<Args>(args)..., allocator);
    }
    else {
      new(&data_[n]) value_type(std::forward<Args>(args)...);
    }
    ++n;
    return data_[n];
  }

  CUDA void pop_back() {
    assert(n > 0);
    data_[n-1].~T();
    --n;
  }

  CUDA NI void resize(size_t count) {
    if(count < n) {
      if constexpr(!std::is_trivially_destructible_v<value_type>) {
        for(size_t i = count; i < n; ++i) {
          data_[i].~T();
        }
      }
      n = count;
    }
    else {
      if(count > cap) {
        reallocate(count);
      }
      for(size_t i = n; i < count; ++i) {
        inplace_new(i);
      }
      n = count;
    }
  }

  CUDA void swap(this_type& other) {
    ::battery::swap(n, other.n);
    ::battery::swap(cap, other.cap);
    ::battery::swap(data_, other.data_);
    ::battery::swap(allocator, other.allocator);
  }

  CUDA NI void print() const {
    for(size_t i = 0; i < n; ++i) {
      ::battery::print(data_[i]);
      if(i < n - 1) {
        printf(", ");
      }
    }
  }
};

template<class T1, class Alloc1, class T2, class Alloc2>
CUDA NI bool operator==(const vector<T1, Alloc1>& lhs, const vector<T2, Alloc2>& rhs) {
  if(lhs.size() != rhs.size()) {
    return false;
  }
  else {
    for(size_t i = 0; i < lhs.size(); ++i) {
      if(!(lhs[i] == rhs[i])) {
        return false;
      }
    }
  }
  return true;
}

template<class T1, class Alloc1, class T2, class Alloc2>
CUDA bool operator!=(const vector<T1, Alloc1>& lhs, const vector<T2, Alloc2>& rhs) {
  return !(lhs == rhs);
}


} // namespace battery

#endif
