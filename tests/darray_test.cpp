// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <memory>
#include "darray.hpp"
#include "allocator.hpp"

template<typename Allocator>
void test_array(const DArray<int, Allocator>& a, size_t size, int elem) {
  EXPECT_EQ(a.size(), size);
  for(int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], elem);
  }
}

TEST(DArray, Constructor) {
  DArray<int, StandardAllocator> a1(3, 2);
  DArray<int, StandardAllocator> a2({2,2,2});
  int d[3] = {2,2,2};
  DArray<int, StandardAllocator> a3(3, d);
  DArray<int, StandardAllocator> a4(a1);
  test_array(a1, 3, 2);
  test_array(a2, 3, 2);
  test_array(a3, 3, 2);
  test_array(a4, 3, 2);
  DArray<int, StandardAllocator> a5(3);
  test_array(a5, 3, 0);
}