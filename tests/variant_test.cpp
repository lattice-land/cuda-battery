// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/variant.hpp"
#include "battery/utility.hpp"
#include "battery/string.hpp"
#include "battery/allocator.hpp"
#include "battery/vector.hpp"

using namespace battery;

class Formula {
  using DataT = variant<char, string<>, vector<Formula>>;
  DataT data;

public:
  Formula(char c): data(DataT::create<0>(c)) {}
  Formula(string<> s): data(DataT::create<1>(s)) {}
  Formula(vector<Formula> f): data(DataT::create<2>(f)) {}

  template<size_t i, typename T>
  void expect(T t) {
    EXPECT_EQ(get<i>(data), t);
    EXPECT_EQ(data.index(), i);
    print();
  }

  void print() const {
    ::print(data);
  }

  friend bool operator==(const Formula& lhs, const Formula& rhs);
};

bool operator==(const Formula& lhs, const Formula& rhs) {
  return lhs.data == rhs.data;
}

TEST(Variant, Constructor) {
  Formula c1('a');
  c1.template expect<0>('a');
  Formula c2('a');
  EXPECT_TRUE(c1 == c2);
  Formula c3("abc");
  c3.template expect<1>(string<>("abc"));
  Formula c4("abc");
  EXPECT_TRUE(c3 == c4);
  EXPECT_FALSE(c1 == c3);
  Formula c5('b');
  EXPECT_FALSE(c1 == c5);
  Formula c6(vector<Formula>(3, Formula('a')));
  c6.template expect<2>(vector<Formula>(3, Formula('a')));
  Formula c7(vector<Formula>(3, Formula('a')));
  EXPECT_TRUE(c6 == c7);
  Formula c8(vector<Formula>(3, Formula('b')));
  EXPECT_FALSE(c6 == c8);
}
