// Copyright 2021 Pierre Talbot

#ifndef STRING_HPP
#define STRING_HPP

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

  /** Allocate a string from `raw_string` using `allocator`. */
  CUDA String(const char* raw_string, const Allocator& alloc = Allocator()):
    data_(strlen(raw_string), raw_string, alloc) {}

  /** Copy constructor with an allocator. */
  template <typename Allocator2>
  CUDA String(const String<Allocator2>& from, const Allocator& alloc = Allocator()):
    data_(from.data_, alloc) {}

  /** Redefine the copy constructor to be sure it calls a constructor with an allocator. */
  CUDA String(const String<Allocator>& from): String(from, Allocator()) {}

  HOST String(const std::string& from, const Allocator& alloc = Allocator()):
    data_(from.size(), from.data(), alloc) {}

  CUDA size_t size() const { return data_.size(); }
  CUDA char& operator[](size_t i) { return data_[i]; }
  CUDA const char& operator[](size_t i) const { return data_[i]; }
  CUDA char* data() { return data_.data(); }
  CUDA const char* data() const { return data_.data(); }
};

#endif
