// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "string.hpp"
#include "allocator.hpp"
#include "utility.hpp"

using namespace battery;

template<typename Allocator>
void test_string(const string<Allocator>& a, const char* expect) {
  EXPECT_EQ(a.size(), battery::strlen(expect));
  for(int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], expect[i]);
  }
}

TEST(String, Constructor) {
  string<StandardAllocator> a1("abc");
  test_string(a1, "abc");
  string<StandardAllocator> a2(a1);
  test_string(a2, "abc");
  std::string test("std::string");
  string<StandardAllocator> a3(test);
  test_string(a3, test.c_str());
  string<StandardAllocator> a4(1);
  a4[0] = '0';
  test_string(a4, "0");
  string<StandardAllocator> a5("");
  test_string(a5, "");
  string<StandardAllocator> a6;
  test_string(a6, "");
  string<StandardAllocator> a7(a1);
  test_string(a7, "abc");
  string<StandardAllocator> a8 = a7;
  test_string(a8, "abc");
}

TEST(String, Equality) {
  string<StandardAllocator> a1("abc");
  string<StandardAllocator> a2(a1);
  string<StandardAllocator> a3("abcd");
  string<StandardAllocator> a4("ab");
  EXPECT_EQ(a1 == a2, true);
  EXPECT_EQ(a2 == a1, true);
  EXPECT_EQ(a1 == a3, false);
  EXPECT_EQ(a3 == a1, false);
  EXPECT_EQ(a1 == a4, false);
  EXPECT_EQ(a3 == a4, false);
  EXPECT_EQ(a4 == a3, false);
  EXPECT_EQ(a3 == a4, false);
  string<StandardAllocator> a5("");
  EXPECT_EQ(a5 == a1, false);
  EXPECT_EQ(a5 == a2, false);
  EXPECT_EQ(a5 == a3, false);
  EXPECT_EQ(a5 == a4, false);
  EXPECT_EQ(a1 == a5, false);
  EXPECT_EQ(a2 == a5, false);
  EXPECT_EQ(a3 == a5, false);
  EXPECT_EQ(a4 == a5, false);
  string<StandardAllocator> a6;
  EXPECT_EQ(a5.size(), 0);
  EXPECT_EQ(a6.size(), 0);
  EXPECT_EQ(a5 == a6, true);
  EXPECT_EQ(a6 == a5, true);
}

TEST(String, Concatenation) {
  string<StandardAllocator> empty;
  string<StandardAllocator> a1("abc");
  string<StandardAllocator> a2("abc");
  string<StandardAllocator> a3("d");
  string<StandardAllocator> a4("dabc");
  EXPECT_EQ(empty + a1, a1);
  EXPECT_EQ(empty + a1, a2);
  EXPECT_EQ(empty + a1, "abc");
  EXPECT_EQ(a1 + empty, "abc");
  EXPECT_EQ(a1 + a1, "abcabc");
  EXPECT_EQ(a1 + a2, "abcabc");
  EXPECT_EQ(a1 + a3, "abcd");
  EXPECT_EQ(a3 + a1, "dabc");
  EXPECT_EQ(a3 + a1, a4);
  EXPECT_EQ(a1 + a3 + a1, "abcdabc");
  EXPECT_EQ("abcdabc", a1 + a3 + a1);
}

TEST(String, FromInt) {
  EXPECT_EQ(string<StandardAllocator>::from_int(0), "0");
  EXPECT_EQ(string<StandardAllocator>::from_int(1), "1");
  EXPECT_EQ(string<StandardAllocator>::from_int(10), "10");
  EXPECT_EQ(string<StandardAllocator>::from_int(100), "100");
  EXPECT_EQ(string<StandardAllocator>::from_int(101), "101");
  EXPECT_EQ(string<StandardAllocator>::from_int(2320194), "2320194");
  EXPECT_EQ(string<StandardAllocator>::from_int(-1), "-1");
  EXPECT_EQ(string<StandardAllocator>::from_int(-10), "-10");
  EXPECT_EQ(string<StandardAllocator>::from_int(-100), "-100");
  EXPECT_EQ(string<StandardAllocator>::from_int(-101), "-101");
  EXPECT_EQ(string<StandardAllocator>::from_int(-2320194), "-2320194");
}
