// Copyright 2023 Pierre Talbot

#include <gtest/gtest.h>
#include "par_map.hpp"

TEST(DEMO, par_map_cpu_test) {
  std::vector<int> original(10000, 50);
  std::vector<int> expected(10000, 100);
  block_par_map(original, [](int x) { return x * 2; });
  EXPECT_EQ(original, expected);
}
