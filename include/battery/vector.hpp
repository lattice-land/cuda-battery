// Copyright 2021 Pierre Talbot

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "utility.hpp"
#include "allocator.hpp"
#include <memory>
#include <initializer_list>
#include <vector>

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

  CUDA void reallocate(size_t new_cap) {
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
    // Free the old data array that has been reallocated.
    for(size_t i = 0; i < n2; ++i) {
      data2[i].~T();
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
      new(&data_[i]) value_type;
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
  CUDA vector(size_t n, const allocator_type& alloc = allocator_type()):
    n(n), cap(n), allocator(alloc), data_(allocate())
  {
      for(size_t i = 0; i < n; ++i) {
        inplace_new(i);
      }
  }

  /** Default constructor.*/
  CUDA vector(const allocator_type& alloc = allocator_type())
   : n(0), cap(0), allocator(alloc), data_(nullptr) {}

  /** Allocate an array of size `n` using `allocator`.
      Initialize the elements of the array with those of the array `from`.
      `Allocator` is scoped, meaning it will be passed to the constructor of `T` if `T(const T&, const Allocator&)` exists, otherwise `T(const T&)` is called.  */
  template <class U>
  CUDA vector(const U* from, size_t n, const allocator_type& alloc = allocator_type())
   : n(n), cap(n), allocator(alloc), data_(allocate())
  {
    for(size_t i = 0; i < n; ++i) {
      inplace_new(i, from[i]);
    }
  }

  /** Copy constructor with an allocator, useful when the vector we copied from was declared using another allocator. */
  template <class U, class Allocator2>
  CUDA vector(const vector<U, Allocator2>& from, const allocator_type& alloc = allocator_type())
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
  CUDA vector(size_t n, const U& from, const allocator_type& alloc = allocator_type())
   : n(n), cap(n), allocator(alloc), data_(allocate())
  {
    for(size_t i = 0; i < n; ++i) {
      inplace_new(i, from);
    }
  }

  CUDA vector(this_type&& other): this_type(other.allocator) {
    swap(other);
  }

  CUDA vector(std::initializer_list<T> init, const Allocator& alloc = Allocator())
   : n(init.size()), cap(init.size()), allocator(alloc), data_(allocate())
  {
    size_t i = 0;
    for(const T& v : init) {
      inplace_new(i, v);
      ++i;
    }
  }

  template <class U, class Alloc2>
  vector(const std::vector<U, Alloc2>& other, const allocator_type& alloc = allocator_type())
   : n(other.size()), cap(other.size()), allocator(alloc), data_(allocate())
  {
    for(size_t i = 0; i < n; ++i) {
      inplace_new(i, other[i]);
    }
  }

  CUDA ~vector() {
    for(size_t i = 0; i < n; ++i) {
      data_[i].~T();
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

  /** Beware that this operator does not free the memory of `this`, the capacity remains unchanged. */
  CUDA this_type& operator=(const this_type& other) {
    reserve(other.size());
    for(size_t i = 0; i < other.n; ++i) {
      if(i < n) {
        data_[i].~T();
      }
      inplace_new(i, other.data_[i]);
    }
    for(size_t i = other.n; i < n; ++i) {
      data_[i].~T();
    }
    n = other.n;
    return *this;
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

  CUDA void clear() {
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

  CUDA void resize(size_t count) {
    if(count < n) {
      for(size_t i = count; i < n; ++i) {
        data_[i].~T();
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

  CUDA void print() const {
    for(size_t i = 0; i < n; ++i) {
      ::battery::print(data_[i]);
      if(i < n - 1) {
        printf(", ");
      }
    }
  }
};

template<class T, class Allocator>
CUDA bool operator==(const vector<T, Allocator>& lhs, const vector<T, Allocator>& rhs) {
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

template<class T, class Allocator>
CUDA bool operator!=(const vector<T, Allocator>& lhs, const vector<T, Allocator>& rhs) {
  return !(lhs == rhs);
}


} // namespace battery

#endif
