// Copyright 2021 Pierre Talbot

#ifndef UNIQUE_PTR_HPP
#define UNIQUE_PTR_HPP

#include "utility.hpp"
#include "allocator.hpp"
#include <memory>

namespace battery {

/** Similar to std::unique_ptr with small differences:
 *   - There is no specialization for arrays (e.g., unique_ptr<T[]>).
     - We rely on an allocator and provide a constructor to build the pointer in place.
    Similarly to vector, the allocator is scoped, meaning it is propagated to the underlying type constructor if it takes one. */
template <class T, class Allocator = StandardAllocator>
class unique_ptr {
public:
  using value_type = T;
  using pointer = T*;
  using allocator_type = Allocator;
  using this_type = unique_ptr<value_type, allocator_type>;
private:
  Allocator allocator;
  T* ptr;
public:
  CUDA unique_ptr(const allocator_type& alloc = allocator_type())
   : allocator(allocator), ptr(nullptr) {}
  CUDA unique_ptr(std::nullptr_t, const allocator_type& alloc = allocator_type())
   : allocator(allocator), ptr(nullptr) {}

  // `ptr` must have been allocated using `allocator_type`.
  CUDA explicit unique_ptr(pointer ptr, const allocator_type& alloc = allocator_type())
   : allocator(allocator), ptr(ptr) {}
  CUDA unique_ptr(this_type&& from) : ptr(from.ptr), allocator(from.allocator) {
    from.ptr = nullptr;
  }
  CUDA unique_ptr(const this_type&) = delete;

  struct unique_tag_t{};

  CUDA unique_ptr(unique_tag_t, const value_type& v, const allocator_type& alloc = allocator_type())
   : allocator(alloc), ptr(static_cast<T*>(allocator.allocate(sizeof(T))))
  {
    if constexpr(std::is_constructible<value_type, const value_type&, const allocator_type&>{}) {
      new(ptr) value_type(v, allocator);
    }
    else {
      new(ptr) value_type(v);
    }
  }

  CUDA unique_ptr(unique_tag_t, value_type&& v, const allocator_type& alloc = allocator_type())
   : allocator(alloc), ptr(static_cast<T*>(allocator.allocate(sizeof(T))))
  {
    if constexpr(std::is_constructible<value_type, value_type&&, const allocator_type&>{}) {
      new(ptr) value_type(std::forward<value_type>(v), allocator);
    }
    else {
      new(ptr) value_type(std::forward<value_type>(v));
    }
  }

  CUDA ~unique_ptr() {
    if(ptr != nullptr) {
      ptr->~T();
      allocator.deallocate(ptr);
      ptr = nullptr;
    }
  }

  CUDA void swap(unique_ptr& other) {
    ::battery::swap(ptr, other.ptr);
    ::battery::swap(allocator, other.allocator);
  }

  CUDA unique_ptr& operator=(unique_ptr&& r) {
    swap(r);
    return *this;
  }

  CUDA unique_ptr& operator=(std::nullptr_t) {
    unique_ptr to_delete(allocator);
    swap(to_delete);
    return *this;
  }

  CUDA pointer release() {
    pointer p = ptr;
    ptr = nullptr;
    return p;
  }

  CUDA void reset(pointer ptr = pointer()) {
    unique_ptr to_swap(ptr, allocator);
    swap(to_swap);
  }

  CUDA pointer get() const {
    return ptr;
  }

  CUDA allocator_type get_allocator() const {
    return allocator;
  }

  CUDA explicit operator bool() const {
    return ptr != nullptr;
  }

  CUDA T& operator*() const {
    assert(bool(ptr));
    return *ptr;
  }

  CUDA pointer operator->() const {
    assert(bool(ptr));
    return ptr;
  }
};

template <class T, class Allocator = StandardAllocator>
CUDA unique_ptr<T, Allocator> allocate_unique(const T& v, const Allocator& alloc = Allocator())
{
  using unique_t = typename unique_ptr<T, Allocator>::unique_tag_t;
  return unique_ptr<T, Allocator>(unique_t{}, v, alloc);
}

template <class T, class Allocator = StandardAllocator>
CUDA unique_ptr<T, Allocator> allocate_unique(T&& v, const Allocator& alloc = Allocator())
{
  using unique_t = typename unique_ptr<T, Allocator>::unique_tag_t;
  return unique_ptr<T, Allocator>(unique_t{}, std::forward<T>(v), alloc);
}

} // namespace battery

#endif
