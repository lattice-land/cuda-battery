// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "vector.hpp"
#include "allocator.hpp"

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
  vector<int> a3(3, d);
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
