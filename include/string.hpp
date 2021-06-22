// Copyright 2021 Pierre Talbot

#ifndef STRING_HPP
#define STRING_HPP

#include <string>
#include "utility.hpp"
#include "darray.hpp"

/** `String` represents a fixed sized array of characters based on `DArray<char>` (see darray.hpp). */
template<typename Allocator>
class String {
  DArray<char, Allocator> data_;
public:
  typedef String<Allocator> this_type;

  /** Allocate a string of size `n` using `allocator`. */
  CUDA String(size_t n, const Allocator& alloc = Allocator()):
    data_(n, alloc) {}

  /** Default constructor. Since the size is 0 and the string cannot be extended, the allocator does not matter. */
  CUDA String(): data_() {}

  /** Allocate a string from `raw_string` using `allocator`. */
  CUDA String(const char* raw_string, const Allocator& alloc = Allocator()):
    data_(strlen(raw_string), raw_string, alloc) {}

  /** Copy constructor with an allocator. */
  template <typename Allocator2>
  CUDA String(const String<Allocator2>& other, const Allocator& alloc = Allocator()):
    data_(other.data_, alloc) {}

  /** Redefine the copy constructor to be sure it calls a constructor with an allocator. */
  CUDA String(const String<Allocator>& other): String(other, Allocator()) {}

  CUDA String(String<Allocator>&& other) = default;
  CUDA String<Allocator>& operator=(String<Allocator> other) {
    data_ = other.data_;
    return *this;
  }

  HOST String(const std::string& other, const Allocator& alloc = Allocator()):
    data_(other.size(), other.data(), alloc) {}

  CUDA size_t size() const { return data_.size(); }
  CUDA char& operator[](size_t i) { return data_[i]; }
  CUDA const char& operator[](size_t i) const { return data_[i]; }
  CUDA char* data() { return data_.data(); }
  CUDA const char* data() const { return data_.data(); }

  template<typename Alloc>
  CUDA friend bool operator==(const String<Alloc>& lhs, const String<Alloc>& rhs);
};

template<typename Allocator>
CUDA bool operator==(const String<Allocator>& lhs, const String<Allocator>& rhs) {
  return lhs.data_ == rhs.data_;
}

#endif
