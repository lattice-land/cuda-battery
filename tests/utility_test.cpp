// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include <climits>
#include <cmath>

using namespace battery;

TEST(Utility, Swap) {
  int i = 0;
  int j = 1;
  battery::swap(i, j);
  EXPECT_EQ(i, 1);
  EXPECT_EQ(j, 0);
  int *ip = &i;
  int *jp = &j;
  battery::swap(ip, jp);
  EXPECT_EQ(ip, &j);
  EXPECT_EQ(jp, &i);
  battery::swap(*ip, *jp);
  EXPECT_EQ(i, 0);
  EXPECT_EQ(j, 1);
}

TEST(Utility, Limits) {
  EXPECT_EQ(limits<int>::neg_inf(), -INT_MAX - 1);
  EXPECT_EQ(limits<int>::inf(), INT_MAX);
  EXPECT_EQ(limits<float>::neg_inf(), -INFINITY);
  EXPECT_EQ(limits<float>::inf(), INFINITY);
}

TEST(Utility, Strlen) {
  EXPECT_EQ(battery::strlen("abc"), 3);
  EXPECT_EQ(battery::strlen(""), 0);
  EXPECT_EQ(battery::strlen("1"), 1);
}
