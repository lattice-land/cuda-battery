// Copyright 2021 Pierre Talbot

#ifndef CUDA_BATTERY_STRING_HPP
#define CUDA_BATTERY_STRING_HPP

#include <string>
#include "utility.hpp"
#include "vector.hpp"

namespace battery {

/** `string` represents a fixed sized array of characters based on `vector<char>`.
    All strings are null-terminated. */
template<class Allocator=standard_allocator>
class string {
  vector<char, Allocator> data_;
public:
  using this_type = string<Allocator>;
  using allocator_type = Allocator;
  using value_type = char;

  template <class Alloc2>
  friend class string;

  /** Allocate a string of size `n` using `allocator`. */
  CUDA string(size_t n, const allocator_type& alloc = allocator_type()):
    data_(n+1, alloc) /* +1 for null-termination */
  {
    data_[n] = '\0'; // In case the user modifies the string.
  }

  CUDA string(const allocator_type& alloc = allocator_type()): string((size_t)0, alloc) {}

  /** Allocate a string from `raw_string` using `allocator`. */
  CUDA string(const char* raw_string, const allocator_type& alloc = allocator_type()):
    data_(raw_string, strlen(raw_string)+1, alloc) {}

  /** Copy constructor with an allocator. */
  template <class Allocator2>
  CUDA string(const string<Allocator2>& other, const allocator_type& alloc = allocator_type()):
    data_(other.data_, alloc) {}

  /** Redefine the copy constructor to be sure it calls a constructor with an allocator. */
  CUDA string(const string<allocator_type>& other): string(other, allocator_type()) {}

  string(string<allocator_type>&& other) = default;
  CUDA string<allocator_type>& operator=(string<allocator_type> other) {
    data_ = other.data_;
    return *this;
  }

  string(const std::string& other, const allocator_type& alloc = allocator_type()):
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

  template <class IntegerType>
  CUDA static this_type from_int(IntegerType x, const allocator_type& alloc = allocator_type()) {
    if(x == 0) { return this_type("0", alloc); }
    size_t s = 0;
    bool neg = x < IntegerType{0};
    if(neg) {
      x = -x;
      s++;
    }
    for(size_t y = x; y > 0; y = y / 10, ++s) {}
    this_type buffer(s, alloc);
    if(neg) {
      buffer[0] = '-';
    }
    for(size_t i = s-1; x > 0; --i) {
      buffer[i] = '0' + (x % 10);
      x = x / 10;
    }
    return std::move(buffer);
  }

  template<class Alloc1, class Alloc2>
  CUDA friend bool operator==(const string<Alloc1>& lhs, const string<Alloc2>& rhs);
};

namespace impl {
  template<class Allocator>
  CUDA string<Allocator> concat(const char* lhs, size_t lhs_len, const char* rhs, size_t rhs_len, const Allocator& alloc) {
    string<Allocator> res(lhs_len + rhs_len, alloc);
    size_t k = 0;
    for(size_t i = 0; i < lhs_len; ++i, ++k) { res[k] = lhs[i]; }
    for(size_t i = 0; i < rhs_len; ++i, ++k) { res[k] = rhs[i]; }
    return std::move(res);
  }
}

template<class Alloc1, class Alloc2>
CUDA bool operator==(const string<Alloc1>& lhs, const string<Alloc2>& rhs) {
  return lhs.size() == rhs.size() && battery::strcmp(lhs.data(), rhs.data()) == 0;
}

template<class Allocator>
CUDA bool operator==(const char* lhs, const string<Allocator>& rhs) {
  return battery::strcmp(lhs, rhs.data()) == 0;
}

template<class Allocator>
CUDA bool operator==(const string<Allocator>& lhs, const char* rhs) {
  return battery::strcmp(lhs.data(), rhs) == 0;
}

template<class Alloc1, class Alloc2>
CUDA bool operator!=(const string<Alloc1>& lhs, const string<Alloc2>& rhs) {
  return !(lhs == rhs);
}

template<class Allocator>
CUDA bool operator!=(const char* lhs, const string<Allocator>& rhs) {
  return !(lhs == rhs);
}

template<class Allocator>
CUDA bool operator!=(const string<Allocator>& lhs, const char* rhs) {
  return !(lhs == rhs);
}

template<class Allocator>
CUDA string<Allocator> operator+(const string<Allocator>& lhs, const string<Allocator>& rhs) {
  return impl::concat(lhs.data(), lhs.size(), rhs.data(), rhs.size(), lhs.get_allocator());
}

template<class Allocator>
CUDA string<Allocator> operator+(const char* lhs, const string<Allocator>& rhs) {
  return impl::concat(lhs, strlen(lhs), rhs.data(), rhs.size(), rhs.get_allocator());
}

template<class Allocator>
CUDA string<Allocator> operator+(const string<Allocator>& lhs, const char* rhs) {
  return impl::concat(lhs.data(), lhs.size(), rhs, strlen(rhs), lhs.get_allocator());
}

} // namespace battery

#endif
