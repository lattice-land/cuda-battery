// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/vector.hpp"
#include "battery/allocator.hpp"
#include "battery/memory.hpp"

using namespace battery;

template<class Allocator>
void test_array(const vector<int, Allocator>& a, size_t size, int elem) {
  EXPECT_EQ(a.size(), size);
  for(int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], elem);
  }
}

TEST(Vector, Constructor) {
  vector<int> a1(3, 2);
  vector<int> a2{2,2,2};
  int d[3] = {2,2,2};
  vector<int> a3(d, 3);
  vector<int> a4(a1);
  test_array(a1, 3, 2);
  test_array(a2, 3, 2);
  test_array(a3, 3, 2);
  test_array(a4, 3, 2);
  vector<int> a5(3);
  test_array(a5, 3, 0);
  vector<int> a6;
  test_array(a6, 0, 0);
  vector<int> a7(std::move(a1));
  test_array(a7, 3, 2);
  vector<int> a8 = a7;
  test_array(a8, 3, 2);
  vector<int> a9;
  a9 = std::move(a8);
  test_array(a9, 3, 2);
}

TEST(Vector, Equality) { // NOTE: Further (indirectly) tested in string_test.
  vector<int> a1{1,2,3};
  vector<int> a2{1,2,3};
  vector<int> a3{2,3,4};
  EXPECT_TRUE(a1 == a2);
  EXPECT_TRUE(a2 == a1);
  EXPECT_FALSE(a1 == a3);
  EXPECT_FALSE(a3 == a1);
}

TEST(Vector, AddAndRemove) {
  vector<int> a1;
  for(int i = 0; i < 100; ++i) {
    a1.push_back(i);
  }
  EXPECT_EQ(a1.size(), 100);
  for(int i = 0; i < 100; ++i) {
    a1.pop_back();
  }
  EXPECT_EQ(a1.size(), 0);
  for(int i = 0; i < 100; ++i) {
    a1.emplace_back(i);
  }
  EXPECT_EQ(a1.size(), 100);
  a1.clear();
  EXPECT_EQ(a1.size(), 0);
}

class NoCopyDefault {
  int x;
  int y;
  public:
    NoCopyDefault() { assert(false); }
    NoCopyDefault(const NoCopyDefault&) { assert(false); }
    NoCopyDefault(int x, int y): x(x), y(y) {}
    NoCopyDefault(NoCopyDefault&&) = default;
};

TEST(Vector, EmplaceBack) {
  vector<NoCopyDefault> a;
  a.reserve(10);
  EXPECT_EQ(a.capacity(), 10);
  EXPECT_EQ(a.size(), 0);
  for(int i = 0; i < 10; ++i) {
    a.emplace_back(i, i*i);
  }
  EXPECT_EQ(a.size(), 10);
  vector<NoCopyDefault> b(std::move(a));
  EXPECT_EQ(b.capacity(), 10);
  EXPECT_EQ(b.size(), 10);
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.capacity(), 0);
  vector<NoCopyDefault> c = std::move(b);
  EXPECT_EQ(c.capacity(), 10);
  EXPECT_EQ(c.size(), 10);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.capacity(), 0);
}

TEST(Vector, ResizeShrink) {
  vector<int> a(100, 5);
  EXPECT_EQ(a.size(), 100);
  for(int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], 5);
  }
  a.resize(150);
  for(int i = 0; i < 100; ++i) {
    EXPECT_EQ(a[i], 5);
  }
  for(int i = 100; i < 150; ++i) {
    EXPECT_EQ(a[i], 0);
  }
  a.push_back(1);
  EXPECT_NE(a.capacity(), a.size());
  EXPECT_EQ(a.capacity(), 150 * 3);
  a.shrink_to_fit();
  EXPECT_EQ(a.capacity(), a.size());
  a.clear();
  EXPECT_EQ(a.size(), 0);
}

TEST(Vector, PoolAlloc) {
  unsigned char mem[100];
  size_t alignment = 10;
  pool_allocator alloc(mem, 100, alignment);
  int wasted_mem = (alignment - (size_t)mem % alignment) % alignment;
  vector<int, pool_allocator> v1{3, alloc};
  EXPECT_EQ(alloc.used(), sizeof(int) * 3 + wasted_mem);
  vector<int, pool_allocator> v2{3, alloc};
  wasted_mem += 8;
  EXPECT_EQ(alloc.used(), sizeof(int) * 6 + wasted_mem);
  alignment = sizeof(int);
  alloc.align_at(alignment);
  wasted_mem += (alignment - (((size_t)mem + alloc.used()) % alignment)) % alignment;
  vector<int, pool_allocator> v3{3, alloc};
  EXPECT_EQ(alloc.used(), sizeof(int) * 9 + wasted_mem);
}

TEST(Vector, STLVectorConstructor) {
  std::vector<std::vector<int>> v1(100, std::vector<int>(100, 0));
  for(int i = 0; i < v1.size(); ++i) {
    for(int j = 0; j < v1[i].size(); ++j) {
      v1[i][j] = i+j*100;
    }
  }
  vector<vector<int, battery::standard_allocator>, battery::standard_allocator> v(v1);
  EXPECT_EQ(v.size(), v1.size());
  for(size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v[i].size(), v1[i].size());
    for(size_t j = 0; j < v[i].size(); ++j) {
      EXPECT_EQ(v[i][j], v1[i][j]);
    }
  }
}
