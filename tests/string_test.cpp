// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "string.hpp"
#include "allocator.hpp"
#include "utility.hpp"

template<typename Allocator>
void test_string(const String<Allocator>& a, const char* expect) {
  EXPECT_EQ(a.size(), battery::strlen(expect));
  for(int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], expect[i]);
  }
}

TEST(String, Constructor) {
  String<StandardAllocator> a1("abc");
  test_string(a1, "abc");
  String<StandardAllocator> a2(a1);
  test_string(a2, "abc");
  std::string test("std::string");
  String<StandardAllocator> a3(test);
  test_string(a3, test.c_str());
  String<StandardAllocator> a4(1);
  a4[0] = '0';
  test_string(a4, "0");
  String<StandardAllocator> a5("");
  test_string(a5, "");
}
