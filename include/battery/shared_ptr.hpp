// Copyright 2021 Pierre Talbot

#ifndef SHARED_PTR_HPP
#define SHARED_PTR_HPP

#include "utility.hpp"
#include "allocator.hpp"

namespace battery {

/** Similar to std::shared_ptr with small differences:
 *   - There is no specialization for arrays (e.g., shared_ptr<T[]>).
     - We rely on an allocator to allocate/deallocate the memory.
     - No support for aliasing constructors.
     - No special thread-safety support.
    Similarly to vector, the allocator is scoped, meaning it is propagated to the underlying type constructor if it takes one. */
template <class T, class Allocator = standard_allocator>
class shared_ptr {
public:
  using element_type = T;
  using pointer = T*;
  using allocator_type = Allocator;
  using this_type = shared_ptr<element_type, allocator_type>;
private:
  Allocator allocator;
  int* count;
  T* ptr;

  template<class Y, class Alloc>
  friend class shared_ptr;

  CUDA int* allocate_counter() {
    int* c = static_cast<int*>(allocator.allocate(sizeof(int)));
    *c = 1;
    return c;
  }

  CUDA void inc_counter() {
    if(count != nullptr) {
      ++(*count);
    }
  }

public:
  CUDA shared_ptr(const allocator_type& alloc = allocator_type())
   : allocator(allocator), count(nullptr), ptr(nullptr) {}
  CUDA shared_ptr(std::nullptr_t, const allocator_type& alloc = allocator_type())
   : allocator(allocator), count(nullptr), ptr(nullptr) {}

  // `ptr` must have been allocated using `allocator_type`.
  template<class Y>
  CUDA explicit shared_ptr(Y* ptr, const allocator_type& alloc = allocator_type())
   : allocator(alloc), count(allocate_counter()), ptr(static_cast<T*>(ptr))
  {
  }

  CUDA shared_ptr(this_type&& from)
   : allocator(from.allocator), ptr(from.ptr), count(from.count)
  {
    from.ptr = nullptr;
    from.count = nullptr;
  }

  template<class Y>
  CUDA shared_ptr(shared_ptr<Y, Allocator>&& from)
   : allocator(from.allocator), ptr(static_cast<T*>(from.ptr)), count(from.count)
  {
    from.ptr = nullptr;
    from.count = nullptr;
  }

  CUDA shared_ptr(const this_type& from)
    : allocator(from.allocator), ptr(from.ptr), count(from.count)
  {
    inc_counter();
  }

  template<class Y>
  CUDA shared_ptr(const shared_ptr<Y, Allocator>& from)
   : allocator(from.allocator), ptr(static_cast<T*>(from.ptr)), count(from.count)
  {
    inc_counter();
  }

  CUDA ~shared_ptr() {
    if(ptr != nullptr) {
      if(*count <= 1) {
        ptr->~T();
        allocator.deallocate(ptr);
        ptr = nullptr;
        allocator.deallocate(count);
        count = nullptr;
      }
      else {
        --(*count);
      }
    }
  }

  CUDA void swap(shared_ptr& other) {
    ::battery::swap(ptr, other.ptr);
    ::battery::swap(count, other.count);
    ::battery::swap(allocator, other.allocator);
  }

  CUDA shared_ptr& operator=(const shared_ptr& r) {
    this_type(r).swap(*this);
    return *this;
  }

  CUDA shared_ptr& operator=(shared_ptr&& r) {
    this_type(std::move(r)).swap(*this);
    return *this;
  }

  template<class Y>
  CUDA shared_ptr& operator=(const shared_ptr<Y, Allocator>& r) {
    this_type(r).swap(*this);
    return *this;
  }

  template<class Y>
  CUDA shared_ptr& operator=(shared_ptr<Y, Allocator>&& r) {
    this_type(std::move(r)).swap(*this);
    return *this;
  }


  CUDA shared_ptr& operator=(std::nullptr_t) {
    this_type(allocator).swap(*this);
    return *this;
  }

  CUDA void reset() {
    this_type(allocator).swap(*this);
  }

  template<class Y>
  CUDA void reset(Y* ptr) {
    this_type(ptr, allocator).swap(*this);
  }

  CUDA pointer get() const {
    return ptr;
  }

  CUDA T& operator*() const {
    assert(bool(ptr));
    return *ptr;
  }

  CUDA pointer operator->() const {
    assert(bool(ptr));
    return ptr;
  }

  CUDA int use_count() const {
    return count == nullptr ? 0 : *count;
  }

  CUDA allocator_type get_allocator() const {
    return allocator;
  }

  CUDA explicit operator bool() const {
    return ptr != nullptr;
  }
};

template<class T, class Alloc, class... Args>
CUDA shared_ptr<T, Alloc> allocate_shared(const Alloc& alloc, Args&&... args) {
  Alloc allocator(alloc);
  T* ptr = static_cast<T*>(allocator.allocate(sizeof(T)));
  assert(ptr != nullptr);
  if constexpr(std::is_constructible<T, Args&&..., const Alloc&>{}) {
    new(ptr) T(std::forward<Args>(args)..., allocator);
  }
  else {
    new(ptr) T(std::forward<Args>(args)...);
  }
  return shared_ptr<T, Alloc>(ptr, allocator);
}

/** Similar to `allocate_shared` but with an default-constructed allocator. */
template<class T, class Alloc, class... Args>
CUDA shared_ptr<T, Alloc> make_shared(Args&&... args) {
  return allocate_shared<T>(Alloc(), std::forward<Args>(args)...);
}

} // namespace battery

#endif
