// Copyright 2021 Pierre Talbot

#ifndef STRING_HPP
#define STRING_HPP

#include <string>
#include "utility.hpp"
#include "vector.hpp"

namespace battery {

/** `String` represents a fixed sized array of characters based on `vector<char>`.
    All strings are null-terminated. */
template<class Allocator>
class String {
  vector<char, Allocator> data_;
public:
  using this_type = String<Allocator>;
  using allocator_type = Allocator;
  using value_type = char;

  template <class Alloc2>
  friend class String;

  /** Allocate a string of size `n` using `allocator`. */
  CUDA String(size_t n, const Allocator& alloc = Allocator()):
    data_(n+1, alloc) /* +1 for null-termination */
  {
    data_[0] = '\0'; // to have `strlen(s.data()) == s.size()`
    data_[n] = '\0'; // In case the user modifies the string.
  }

  CUDA String(const Allocator& alloc = Allocator()): String((size_t)0, alloc) {}

  /** Allocate a string from `raw_string` using `allocator`. */
  CUDA String(const char* raw_string, const Allocator& alloc = Allocator()):
    data_(raw_string, strlen(raw_string)+1, alloc) {}

  /** Copy constructor with an allocator. */
  template <class Allocator2>
  CUDA String(const String<Allocator2>& other, const Allocator& alloc = Allocator()):
    data_(other.data_, alloc) {}

  /** Redefine the copy constructor to be sure it calls a constructor with an allocator. */
  CUDA String(const String<Allocator>& other): String(other, Allocator()) {}

  String(String<Allocator>&& other) = default;
  CUDA String<Allocator>& operator=(String<Allocator> other) {
    data_ = other.data_;
    return *this;
  }

  HOST String(const std::string& other, const Allocator& alloc = Allocator()):
    data_(other.data(), other.size()+1, alloc) {}

  CUDA allocator_type get_allocator() const { return data_.get_allocator(); }
  CUDA size_t size() const { return data_.size() == 0 ? 0 : (data_.size() - 1); }
  CUDA char& operator[](size_t i) { assert(i < size()); return data_[i]; }
  CUDA const char& operator[](size_t i) const { assert(i < size()); return data_[i]; }

  CUDA char* data() { return data_.data(); }
  CUDA const char* data() const { return data_.data(); }

  CUDA void print() const {
    printf("%s", data());
  }

  template<typename Alloc>
  CUDA friend bool operator==(const String<Alloc>& lhs, const String<Alloc>& rhs);
};

namespace impl {
  template<class Allocator>
  CUDA String<Allocator> concat(const char* lhs, size_t lhs_len, const char* rhs, size_t rhs_len, Allocator alloc) {
    String<Allocator> res(lhs_len + rhs_len, alloc);
    int k = 0;
    for(int i = 0; i < lhs_len; ++i, ++k) { res[k] = lhs[i]; }
    for(int i = 0; i < rhs_len; ++i, ++k) { res[k] = rhs[i]; }
    return std::move(res);
  }
}

template<class Allocator>
CUDA bool operator==(const String<Allocator>& lhs, const String<Allocator>& rhs) {
  return lhs.size() == rhs.size() && battery::strcmp(lhs.data(), rhs.data()) == 0;
}

template<class Allocator>
CUDA bool operator==(const char* lhs, const String<Allocator>& rhs) {
  return battery::strcmp(lhs, rhs.data()) == 0;
}

template<class Allocator>
CUDA bool operator==(const String<Allocator>& lhs, const char* rhs) {
  return battery::strcmp(lhs.data(), rhs) == 0;
}

template<class Allocator>
CUDA bool operator!=(const String<Allocator>& lhs, const String<Allocator>& rhs) {
  return !(lhs == rhs);
}

template<class Allocator>
CUDA bool operator!=(const char* lhs, const String<Allocator>& rhs) {
  return !(lhs == rhs);
}

template<class Allocator>
CUDA bool operator!=(const String<Allocator>& lhs, const char* rhs) {
  return !(lhs == rhs);
}

template<class Allocator>
CUDA String<Allocator> operator+(const String<Allocator>& lhs, const String<Allocator>& rhs) {
  return impl::concat(lhs.data(), lhs.size(), rhs.data(), rhs.size(), lhs.get_allocator());
}

template<class Allocator>
CUDA String<Allocator> operator+(const char* lhs, const String<Allocator>& rhs) {
  return impl::concat(lhs, strlen(lhs), rhs.data(), rhs.size(), rhs.get_allocator());
}

template<class Allocator>
CUDA String<Allocator> operator+(const String<Allocator>& lhs, const char* rhs) {
  return impl::concat(lhs.data(), lhs.size(), rhs, strlen(rhs), lhs.get_allocator());
}

} // namespace battery

#endif
