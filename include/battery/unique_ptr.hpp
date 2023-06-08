// Copyright 2021 Pierre Talbot

#ifndef UNIQUE_PTR_HPP
#define UNIQUE_PTR_HPP

#include "utility.hpp"
#include "allocator.hpp"

namespace battery {

/** Similar to std::unique_ptr with small differences:
 *   - There is no specialization for arrays (e.g., unique_ptr<T[]>).
     - We rely on an allocator and provide a function `allocate_unique` to build the pointer in place.
    Similarly to vector, the allocator is scoped, meaning it is propagated to the underlying type constructor if it takes one. */
template <class T, class Allocator = standard_allocator>
class unique_ptr {
public:
  using element_type = T;
  using pointer = T*;
  using allocator_type = Allocator;
  using this_type = unique_ptr<element_type, allocator_type>;
private:
  Allocator allocator;
  T* ptr;

  template<class U, class Alloc>
  friend class unique_ptr;
public:
  CUDA unique_ptr(const allocator_type& allocator = allocator_type())
   : allocator(allocator), ptr(nullptr) {}
  CUDA unique_ptr(std::nullptr_t, const allocator_type& allocator = allocator_type())
   : allocator(allocator), ptr(nullptr) {}

  // `ptr` must have been allocated using `allocator_type`.
  CUDA explicit unique_ptr(pointer ptr, const allocator_type& allocator = allocator_type())
   : allocator(allocator), ptr(ptr) {}

  CUDA unique_ptr(this_type&& from) : ptr(from.ptr), allocator(from.allocator) {
    from.ptr = nullptr;
  }

  template<class U>
  CUDA unique_ptr(unique_ptr<U, Allocator>&& from)
   : ptr(static_cast<T*>(from.ptr)), allocator(from.allocator)
  {
    from.ptr = nullptr;
  }

  CUDA unique_ptr(const this_type&) = delete;

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
    this_type(std::move(r)).swap(*this);
    return *this;
  }

  template<class U>
  CUDA unique_ptr& operator=(unique_ptr<U, Allocator>&& r) {
    this_type(std::move(r)).swap(*this);
    return *this;
  }

  CUDA unique_ptr& operator=(std::nullptr_t) {
    this_type(allocator).swap(*this);
    return *this;
  }

  CUDA pointer release() {
    pointer p = ptr;
    ptr = nullptr;
    return p;
  }

  CUDA void reset(pointer ptr = pointer()) {
    this_type(ptr, allocator).swap(*this);
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

template<class T, class Alloc, class... Args>
CUDA unique_ptr<T, Alloc> allocate_unique(const Alloc& alloc, Args&&... args) {
  Alloc allocator(alloc);
  T* ptr = static_cast<T*>(allocator.allocate(sizeof(T)));
  assert(ptr != nullptr);
  if constexpr(std::is_constructible<T, Args&&..., const Alloc&>{}) {
    new(ptr) T(std::forward<Args>(args)..., allocator);
  }
  else {
    new(ptr) T(std::forward<Args>(args)...);
  }
  return unique_ptr<T, Alloc>(ptr, allocator);
}

/** Similar to `allocate_unique` but with an default-constructed allocator. */
template<class T, class Alloc, class... Args>
CUDA unique_ptr<T, Alloc> make_unique(Args&&... args) {
  return allocate_unique<T>(Alloc(), std::forward<Args>(args)...);
}

} // namespace battery

#endif
