// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "bitset.hpp"
#include "memory.hpp"
#include <climits>
#include <cmath>

using namespace battery;

using Bitset1 = Bitset<1, Memory<StandardAllocator>>;
using Bitset10 = Bitset<10, Memory<StandardAllocator>>;
using Bitset64 = Bitset<64, Memory<StandardAllocator>>;
using Bitset70 = Bitset<70, Memory<StandardAllocator>>;
using Bitset512 = Bitset<512, Memory<StandardAllocator>>;
using Bitset1001 = Bitset<1001, Memory<StandardAllocator>>;

#define TEST_ALL(fun) \
  fun<Bitset1>(); \
  fun<Bitset10>(); \
  fun<Bitset64>(); \
  fun<Bitset70>(); \
  fun<Bitset512>(); \
  fun<Bitset1001>();

template<class B>
void test_default_constructor() {
  const B b;
  EXPECT_TRUE(b.none());
  EXPECT_FALSE(b.any());
  EXPECT_FALSE(b.all());
  EXPECT_EQ(b.count(), 0);
  for(int i = 0; i < b.size(); ++i) {
    EXPECT_FALSE(b.test(i));
  }
}

template<class B>
void test_string_constructor() {
  const B b1("1");
  EXPECT_FALSE(b1.none());
  EXPECT_TRUE(b1.any());
  EXPECT_EQ(b1.all(), b1.size() == b1.count());
  EXPECT_EQ(b1.count(), 1);
  EXPECT_TRUE(b1.test(0));
  for(int i = 1; i < b1.size(); ++i) {
    EXPECT_FALSE(b1.test(i));
  }
  String<StandardAllocator> bitset_str(b1.size());
  for(int i = 0; i < bitset_str.size(); ++i) {
    bitset_str[i] = '1';
  }
  const B b2(bitset_str.data());
  EXPECT_FALSE(b2.none());
  EXPECT_TRUE(b2.any());
  EXPECT_EQ(b2.all(), b2.size() == b2.count());
  EXPECT_EQ(b2.count(), b2.size());
  for(int i = 0; i < b2.size(); ++i) {
    EXPECT_TRUE(b2.test(i));
  }
  const B b3(b2);
  EXPECT_FALSE(b3.none());
  EXPECT_TRUE(b3.any());
  EXPECT_EQ(b3.all(), b3.size() == b3.count());
  EXPECT_EQ(b3.count(), b3.size());
  for(int i = 0; i < b3.size(); ++i) {
    EXPECT_TRUE(b3.test(i));
  }
}

TEST(Bitset, Constructor) {
  TEST_ALL(test_default_constructor);
  TEST_ALL(test_string_constructor);
}

template<class B>
void test_set_and_test() {
  B b;
  b.set();
  EXPECT_EQ(B::ones(), b);
  EXPECT_NE(B::zeroes(), b);
  b.reset();
  EXPECT_NE(B::ones(), b);
  EXPECT_EQ(B::zeroes(), b);

  for(int i = 0; i < b.size(); i += 2) {
    b.set(i);
  }
  for(int i = 0; i < b.size(); ++i) {
    EXPECT_EQ(b.test(i), i % 2 == 0);
  }
}

TEST(Bitset, SetAndTest) {
  TEST_ALL(test_set_and_test);
}

TEST(Bitset, Flip) {
  Bitset<19, Memory<StandardAllocator>, unsigned char> b1("0000001000101010111");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b2("1111110111010101000");
  EXPECT_EQ(b1.count(), b2.size() - b2.count());
  EXPECT_EQ(b1.size() - b1.count(), b2.count());
  EXPECT_NE(b1, b2);
  EXPECT_EQ(~b1, b2);
  EXPECT_EQ(b1, ~b2);
  EXPECT_NE(~b1, ~b2);
  b1.flip();
  EXPECT_EQ(b1, b2);
}

TEST(Bitset, SetOperations) {
  Bitset<19, Memory<StandardAllocator>, unsigned char> zero;
  Bitset<19, Memory<StandardAllocator>, unsigned char> one("0000000000000000001");
  Bitset<19, Memory<StandardAllocator>, unsigned char> two("0000000000000000011");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b1("1000000011000000000");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b2("1000000011000000011");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b3("0001000011000010000");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b4("0000000011000000000");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b5("1001000011000010000");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b6("1001000011000010011");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b7("1001000000000010000");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b8("1001000000000010011");
  Bitset<19, Memory<StandardAllocator>, unsigned char> all;
  all.set();
  // Intersection

  // Zero
  EXPECT_EQ(one & zero, zero);
  EXPECT_EQ(b1 & zero, zero);
  EXPECT_EQ(all & zero, zero);
  EXPECT_EQ(zero & one, zero);
  EXPECT_EQ(zero & b1, zero);
  EXPECT_EQ(zero & all, zero);

  // All
  EXPECT_EQ(one & all, one);
  EXPECT_EQ(b1 & all, b1);
  EXPECT_EQ(all & all, all);
  EXPECT_EQ(all & one, one);
  EXPECT_EQ(all & b1, b1);
  EXPECT_EQ(all & all, all);

  // Any
  EXPECT_EQ(b2 & b1, b1);
  EXPECT_EQ(b1 & b2, b1);
  EXPECT_EQ(b1 & b3, b4);
  EXPECT_EQ(b2 & b3, b4);
  EXPECT_EQ(one & b4, zero);

  // Union

  // Zero
  EXPECT_EQ(one | zero, one);
  EXPECT_EQ(b1 | zero, b1);
  EXPECT_EQ(all | zero, all);
  EXPECT_EQ(zero | one, one);
  EXPECT_EQ(zero | b1, b1);
  EXPECT_EQ(zero | all, all);

  // All
  EXPECT_EQ(one | all, all);
  EXPECT_EQ(b1 | all, all);
  EXPECT_EQ(all | all, all);
  EXPECT_EQ(all | one, all);
  EXPECT_EQ(all | b1, all);
  EXPECT_EQ(all | all, all);

  // Any
  EXPECT_EQ(b2 | b1, b2);
  EXPECT_EQ(b1 | b2, b2);
  EXPECT_EQ(b1 | b3, b5);
  EXPECT_EQ(b2 | b3, b6);
  EXPECT_EQ(b3 | b1, b5);

  // XOR

  // Zero
  EXPECT_EQ(one ^ zero, one);
  EXPECT_EQ(b1 ^ zero, b1);
  EXPECT_EQ(all ^ zero, all);
  EXPECT_EQ(zero ^ one, one);
  EXPECT_EQ(zero ^ b1, b1);
  EXPECT_EQ(zero ^ all, all);

  // All
  EXPECT_EQ(one ^ all, ~one);
  EXPECT_EQ(b1 ^ all, ~b1);
  EXPECT_EQ(all ^ all, zero);
  EXPECT_EQ(all ^ one, ~one);
  EXPECT_EQ(all ^ b1, ~b1);
  EXPECT_EQ(all ^ all, zero);

  // Any
  EXPECT_EQ(b2 ^ b1, two);
  EXPECT_EQ(b1 ^ b2, two);
  EXPECT_EQ(b1 ^ b3, b7);
  EXPECT_EQ(b2 ^ b3, b8);
  EXPECT_EQ(b3 ^ b1, b7);
  EXPECT_EQ(b3 ^ b2, b8);
}

TEST(Bitset, BitCountingOperations) {
  Bitset<19, Memory<StandardAllocator>, unsigned char> zero;
  Bitset<19, Memory<StandardAllocator>, unsigned char> one("0000000000000000001");
  Bitset<19, Memory<StandardAllocator>, unsigned char> two("0000000000000000011");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b1("1000000011000000000");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b2("1000000011000000011");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b3("1111111100011111111");
  Bitset<19, Memory<StandardAllocator>, unsigned char> b4("1111111110111111111");
  Bitset<19, Memory<StandardAllocator>, unsigned char> all;
  all.set();

  EXPECT_EQ(zero.countl_zero(), zero.size());
  EXPECT_EQ(zero.countr_zero(), zero.size());
  EXPECT_EQ(zero.countl_one(), 0);
  EXPECT_EQ(zero.countr_one(), 0);

  EXPECT_EQ(all.countl_zero(), 0);
  EXPECT_EQ(all.countr_zero(), 0);
  EXPECT_EQ(all.countl_one(), all.size());
  EXPECT_EQ(all.countr_one(), all.size());

  EXPECT_EQ(one.countl_zero(), 18);
  EXPECT_EQ(one.countr_zero(), 0);
  EXPECT_EQ(one.countl_one(), 0);
  EXPECT_EQ(one.countr_one(), 1);

  EXPECT_EQ(two.countl_zero(), 17);
  EXPECT_EQ(two.countr_zero(), 0);
  EXPECT_EQ(two.countl_one(), 0);
  EXPECT_EQ(two.countr_one(), 2);

  EXPECT_EQ(b1.countl_zero(), 0);
  EXPECT_EQ(b1.countr_zero(), 9);
  EXPECT_EQ(b1.countl_one(), 1);
  EXPECT_EQ(b1.countr_one(), 0);

  EXPECT_EQ(b2.countl_zero(), 0);
  EXPECT_EQ(b2.countr_zero(), 0);
  EXPECT_EQ(b2.countl_one(), 1);
  EXPECT_EQ(b2.countr_one(), 2);

  EXPECT_EQ(b3.countl_zero(), 0);
  EXPECT_EQ(b3.countr_zero(), 0);
  EXPECT_EQ(b3.countl_one(), 8);
  EXPECT_EQ(b3.countr_one(), 8);

  EXPECT_EQ(b4.countl_zero(), 0);
  EXPECT_EQ(b4.countr_zero(), 0);
  EXPECT_EQ(b4.countl_one(), 9);
  EXPECT_EQ(b4.countr_one(), 9);
}
