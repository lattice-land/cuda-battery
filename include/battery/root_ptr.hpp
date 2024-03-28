// Copyright 2024 Pierre Talbot

#ifndef CUDA_BATTERY_ROOT_PTR_HPP
#define CUDA_BATTERY_ROOT_PTR_HPP

#include "utility.hpp"
#include "allocator.hpp"

/** \file root_ptr.hpp
 This smart pointer is a variant of shared_ptr where only the first instance has the ownership of the object.
 This is intended to be used with hierarchical data structures where the root of the hierarchy is the owner of the data, and the root is deleted after the children.
 A use-case in GPU programming is to distribute shared data over blocks; using shared_ptr would lead to data races since the class is not thread-safe; with root_ptr, only one block is responsible for the deletion.
*/

namespace battery {

template <class T, class Allocator = standard_allocator>
class root_ptr {
public:
  using element_type = T;
  using pointer = T*;
  using allocator_type = Allocator;
  using this_type = root_ptr<element_type, allocator_type>;
private:
  Allocator allocator;
  T* ptr;
  bool root;

  template<class U, class Alloc>
  friend class root_ptr;
public:
  CUDA root_ptr(const allocator_type& allocator = allocator_type())
   : allocator(allocator), ptr(nullptr), root(true) {}
  CUDA root_ptr(std::nullptr_t, const allocator_type& allocator = allocator_type())
   : allocator(allocator), ptr(nullptr), root(true) {}

  // `ptr` must have been allocated using `allocator_type`.
  CUDA explicit root_ptr(pointer ptr, const allocator_type& allocator = allocator_type())
   : allocator(allocator), ptr(ptr), root(true) {}

  CUDA root_ptr(this_type&& from) : ptr(from.ptr), allocator(from.allocator), root(from.root) {
    from.ptr = nullptr;
  }

  template<class U>
  CUDA root_ptr(root_ptr<U, Allocator>&& from)
   : ptr(static_cast<T*>(from.ptr)), allocator(from.allocator), root(from.root)
  {
    from.ptr = nullptr;
  }

  CUDA root_ptr(const this_type& other)
   : ptr(other.ptr), allocator(other.allocator), root(false)
  {}

  CUDA NI ~root_ptr() {
    if(root && ptr != nullptr) {
      ptr->~T();
      allocator.deallocate(ptr);
      ptr = nullptr;
    }
  }

  CUDA void swap(root_ptr& other) {
    ::battery::swap(ptr, other.ptr);
    ::battery::swap(allocator, other.allocator);
    ::battery::swap(root, other.root);
  }

  CUDA root_ptr& operator=(root_ptr&& r) {
    this_type(std::move(r)).swap(*this);
    return *this;
  }

  CUDA root_ptr& operator=(const root_ptr& r) {
    this_type(r).swap(*this);
    root = false;
    return *this;
  }

  template<class U>
  CUDA root_ptr& operator=(root_ptr<U, Allocator>&& r) {
    this_type(std::move(r)).swap(*this);
    return *this;
  }

  CUDA root_ptr& operator=(std::nullptr_t) {
    this_type(allocator).swap(*this);
    return *this;
  }

  CUDA pointer release() {
    pointer p = ptr;
    ptr = nullptr;
    root = true;
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

  CUDA bool is_root() const {
    return root;
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
CUDA NI root_ptr<T, Alloc> allocate_root(const Alloc& alloc, Args&&... args) {
  Alloc allocator(alloc);
  T* ptr = static_cast<T*>(allocator.allocate(sizeof(T)));
  assert(ptr != nullptr);
  if constexpr(std::is_constructible<T, Args&&..., const Alloc&>{}) {
    new(ptr) T(std::forward<Args>(args)..., allocator);
  }
  else {
    new(ptr) T(std::forward<Args>(args)...);
  }
  return root_ptr<T, Alloc>(ptr, allocator);
}

/** Similar to `allocate_root` but with an default-constructed allocator. */
template<class T, class Alloc, class... Args>
CUDA root_ptr<T, Alloc> make_root(Args&&... args) {
  return allocate_root<T>(Alloc(), std::forward<Args>(args)...);
}

}
#endif
