// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "utility.hpp"
#include <climits>
#include <cmath>

using namespace battery;

TEST(Utility, Swap) {
  int i = 0;
  int j = 1;
  impl::swap(i, j);
  EXPECT_EQ(i, 1);
  EXPECT_EQ(j, 0);
  int *ip = &i;
  int *jp = &j;
  impl::swap(ip, jp);
  EXPECT_EQ(ip, &j);
  EXPECT_EQ(jp, &i);
  impl::swap(*ip, *jp);
  EXPECT_EQ(i, 0);
  EXPECT_EQ(j, 1);
}

TEST(Utility, Limits) {
  EXPECT_EQ(Limits<int>::bot(), -INT_MAX - 1);
  EXPECT_EQ(Limits<int>::top(), INT_MAX);
  EXPECT_EQ(Limits<float>::bot(), -INFINITY);
  EXPECT_EQ(Limits<float>::top(), INFINITY);
}

TEST(Utility, Strlen) {
  EXPECT_EQ(impl::strlen("abc"), 3);
  EXPECT_EQ(impl::strlen(""), 0);
  EXPECT_EQ(impl::strlen("1"), 1);
}
