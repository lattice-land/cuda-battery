// Copyright 2021 Pierre Talbot
// Based on https://gist.github.com/calebh/fd00632d9c616d4b0c14e7c2865f3085
// and on https://gist.github.com/tibordp/6909880

#ifndef VARIANT_HPP
#define VARIANT_HPP

#include <type_traits>
#include "utility.hpp"

namespace battery {
namespace impl {

template<size_t n, typename... Ts>
struct variant_helper_rec;

template<size_t n, typename F, typename... Ts>
struct variant_helper_rec<n, F, Ts...>
{
  CUDA inline static void destroy(size_t id, void* data)
  {
    if (n == id) {
      reinterpret_cast<F*>(data)->~F();
    } else {
      variant_helper_rec<n + 1, Ts...>::destroy(id, data);
    }
  }

  CUDA inline static void move(size_t id, void* from, void* to)
  {
    if (n == id) {
      new (to) F(std::move(*reinterpret_cast<F*>(from)));
    } else {
      variant_helper_rec<n + 1, Ts...>::move(id, from, to);
    }
  }

  CUDA inline static void copy(size_t id, const void* from, void* to)
  {
    if (n == id) {
      new (to) F(*reinterpret_cast<const F*>(from));
    } else {
      variant_helper_rec<n + 1, Ts...>::copy(id, from, to);
    }
  }

  CUDA inline static bool equals(size_t id, const void* a, const void* b)
  {
    if (n == id) {
      return *reinterpret_cast<const F*>(a) == *reinterpret_cast<const F*>(b);
    } else {
      return variant_helper_rec<n + 1, Ts...>::equals(id, a, b);
    }
  }

  CUDA inline static void print(size_t id, const void* a)
  {
    if (n == id) {
      ::battery::print(*reinterpret_cast<const F*>(a));
    } else {
      variant_helper_rec<n + 1, Ts...>::print(id, a);
    }
  }
};

template<size_t n> struct variant_helper_rec<n> {
  CUDA inline static void destroy(size_t id, void* data) { }
  CUDA inline static void move(size_t old_t, void* from, void* to) { }
  CUDA inline static void copy(size_t old_t, const void* from, void* to) { }
  CUDA inline static bool equals(size_t id, const void* a, const void* b) { return false; }
  CUDA inline static void print(size_t id, const void* a) {}
};

template<typename... Ts>
struct variant_helper {
  CUDA inline static void destroy(size_t id, void* data) {
    variant_helper_rec<0, Ts...>::destroy(id, data);
  }

  CUDA inline static void move(size_t id, void* from, void* to) {
    variant_helper_rec<0, Ts...>::move(id, from, to);
  }

  CUDA inline static void copy(size_t id, const void* from, void* to) {
    variant_helper_rec<0, Ts...>::copy(id, from, to);
  }

  CUDA inline static bool equals(size_t id, const void* a, const void* b) {
    return variant_helper_rec<0, Ts...>::equals(id, a, b);
  }

  CUDA inline static void print(size_t id, const void* a) {
    return variant_helper_rec<0, Ts...>::print(id, a);
  }
};

template<> struct variant_helper<> {
  CUDA inline static void destroy(size_t id, void* data) { }
  CUDA inline static void move(size_t old_t, void* from, void* to) { }
  CUDA inline static void copy(size_t old_t, const void* from, void* to) { }
  CUDA inline static bool equals(size_t id, const void* a, const void* b) { return false; }
  CUDA inline static void print(size_t id, const void* a) { }
};

template<typename F>
struct variant_helper_static;

template<typename F>
struct variant_helper_static {
  CUDA inline static void move(void* from, void* to) {
    new (to) F(std::move(*reinterpret_cast<F*>(from)));
  }

  CUDA inline static void copy(const void* from, void* to) {
    new (to) F(*reinterpret_cast<const F*>(from));
  }
};

// Given a size_t i, selects the ith type from the list of item types
template<size_t i, typename... Items>
struct variant_alternative;

template<typename HeadItem, typename... TailItems>
struct variant_alternative<0, HeadItem, TailItems...>
{
  using type = HeadItem;
};

template<size_t i, typename HeadItem, typename... TailItems>
struct variant_alternative<i, HeadItem, TailItems...>
{
  using type = typename variant_alternative<i - 1, TailItems...>::type;
};

} // namespace impl

template<typename... Ts>
struct Variant {
private:
  using data_t = typename std::aligned_union<1, Ts...>::type;
  using helper_t = impl::variant_helper<Ts...>;

  template<size_t i>
  using alternative = typename impl::variant_alternative<i, Ts...>::type;

  size_t variant_id;
  data_t data;

  CUDA Variant(size_t id) : variant_id(id) {}

  template<size_t i>
  CUDA alternative<i>& get()
  {
    assert(variant_id == i);
    return *reinterpret_cast<alternative<i>*>(&data);
  }

  template<size_t i>
  CUDA const alternative<i>& get() const
  {
    assert(variant_id == i);
    return *reinterpret_cast<const alternative<i>*>(&data);
  }

public:
  template<size_t i>
  CUDA static Variant create(alternative<i>& value)
  {
    Variant ret(i);
    impl::variant_helper_static<alternative<i>>::copy(&value, &ret.data);
    return ret;
  }

  template<size_t i>
  CUDA static Variant create(alternative<i>&& value) {
    Variant ret(i);
    impl::variant_helper_static<alternative<i>>::move(&value, &ret.data);
    return ret;
  }

  CUDA Variant(const Variant<Ts...>& from) : variant_id(from.variant_id)
  {
    helper_t::copy(from.variant_id, &from.data, &data);
  }

  CUDA Variant(Variant<Ts...>&& from) : variant_id(from.variant_id)
  {
    helper_t::move(from.variant_id, &from.data, &data);
  }

  CUDA Variant<Ts...>& operator= (Variant<Ts...>& rhs)
  {
    helper_t::destroy(variant_id, &data);
    variant_id = rhs.variant_id;
    helper_t::copy(rhs.variant_id, &rhs.data, &data);
    return *this;
  }

  CUDA Variant<Ts...>& operator= (Variant<Ts...>&& rhs)
  {
    helper_t::destroy(variant_id, &data);
    variant_id = rhs.variant_id;
    helper_t::move(rhs.variant_id, &rhs.data, &data);
    return *this;
  }

  CUDA size_t index() const {
    return variant_id;
  }

  template<size_t i>
  CUDA void set(alternative<i>& value)
  {
    helper_t::destroy(variant_id, &data);
    variant_id = i;
    impl::variant_helper_static<alternative<i>>::copy(&value, &data);
  }

  template<size_t i>
  CUDA void set(alternative<i>&& value)
  {
    helper_t::destroy(variant_id, &data);
    variant_id = i;
    impl::variant_helper_static<alternative<i>>::move(&value, &data);
  }

  CUDA ~Variant() {
    helper_t::destroy(variant_id, &data);
  }

  CUDA void print() const {
    helper_t::print(variant_id, &data);
  }

  template<typename... Us>
  CUDA friend bool operator==(const Variant<Us...>& lhs, const Variant<Us...>& rhs);
  template<size_t i, typename... Us>
  CUDA friend typename impl::variant_alternative<i, Us...>::type& get(Variant<Us...>& v);
  template<size_t i, typename... Us>
  CUDA friend const typename impl::variant_alternative<i, Us...>::type& get(const Variant<Us...>& v);
};

template<size_t i, typename... Ts>
CUDA typename impl::variant_alternative<i, Ts...>::type& get(Variant<Ts...>& v)
{
  return v.template get<i>();
}

template<size_t i, typename... Ts>
CUDA const typename impl::variant_alternative<i, Ts...>::type& get(const Variant<Ts...>& v)
{
  return v.template get<i>();
}

template<typename... Ts>
CUDA bool operator==(const Variant<Ts...>& lhs, const Variant<Ts...>& rhs) {
  if(lhs.index() == rhs.index()) {
    return impl::variant_helper<Ts...>::equals(lhs.index(), &lhs.data, &rhs.data);
  }
  else {
    return false;
  }
}

template<typename... Ts>
CUDA bool operator!=(const Variant<Ts...>& lhs, const Variant<Ts...>& rhs) {
  return !(lhs == rhs);
}

} // namespace battery

#endif