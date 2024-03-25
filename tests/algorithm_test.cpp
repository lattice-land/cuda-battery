// Copyright 2024 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/algorithm.hpp"

using namespace battery;

TEST(Algorithm, Quicksort) {
  vector<int> v = {3, 2, 1, 4, 5};
  sort(v, [](int x, int y) { return x < y; });
  for(int i = 1; i <= 5; ++i) {
    EXPECT_EQ(v[i - 1], i);
  }
  sort(v, [](int x, int y) { return x > y; });
  for(int i = 0; i < 5; ++i) {
    EXPECT_EQ(v[i], 5-i);
  }
  sort(v, [](int x, int y) { return x > y; });
  for(int i = 0; i < 5; ++i) {
    EXPECT_EQ(v[i], 5-i);
  }
}