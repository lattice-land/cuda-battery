// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/dynamic_bitset.hpp"
#include "battery/bitset.hpp"
#include "battery/memory.hpp"
#include <climits>
#include <cmath>

using namespace battery;

using DynBitset = dynamic_bitset<local_memory>;
using Bitset1 = bitset<1, local_memory>;
using Bitset10 = bitset<10, local_memory>;
using Bitset64 = bitset<64, local_memory>;
using Bitset70 = bitset<70, local_memory>;
using Bitset512 = bitset<512, local_memory>;
using Bitset1001 = bitset<1001, local_memory>;

#define TEST_ALL_STATIC(fun) \
  fun<Bitset1>(); \
  fun<Bitset10>(); \
  fun<Bitset64>(); \
  fun<Bitset70>(); \
  fun<Bitset512>(); \
  fun<Bitset1001>();

#define TEST_ALL(fun) \
  fun<DynBitset>(); \
  TEST_ALL_STATIC(fun);

template<class B>
void test_default_constructor() {
  const B b;
  EXPECT_TRUE(b.none());
  EXPECT_FALSE(b.any());
  EXPECT_FALSE(b.size() == 0 ? !b.all() : b.all());
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
  string<> bitset_str(b1.size());
  for(int i = 0; i < bitset_str.size(); ++i) {
    bitset_str[i] = '1';
  }
  const B b2(bitset_str.data());
  EXPECT_FALSE(b2.none());
  EXPECT_TRUE(b2.any());
  EXPECT_EQ(b2.all(), b2.size() == b2.count());
  EXPECT_EQ(b2.count(), b1.size());
  for(int i = 0; i < b1.size(); ++i) {
    EXPECT_TRUE(b2.test(i));
  }
  const B b3(b2);
  EXPECT_FALSE(b3.none());
  EXPECT_TRUE(b3.any());
  EXPECT_EQ(b3.all(), b3.size() == b3.count());
  EXPECT_EQ(b3.count(), b1.size());
  for(int i = 0; i < b1.size(); ++i) {
    EXPECT_TRUE(b3.test(i));
  }
  const B b4("");
  EXPECT_TRUE(b4.none());
  EXPECT_FALSE(b4.any());
  EXPECT_EQ(b4.all(), b4.size() == b4.count());
  EXPECT_EQ(b4.count(), 0);
  for(int i = 0; i < b4.size(); ++i) {
    EXPECT_FALSE(b4.test(i));
  }
}

TEST(Bitset, Constructor) {
  TEST_ALL(test_default_constructor);
  TEST_ALL(test_string_constructor);
}

void test_range(int s, int e) {
  DynBitset b(s, e);
  EXPECT_EQ(b.count(), e-s+1);
  int pos = 0;
  for(; pos < s; ++pos) {
    EXPECT_FALSE(b.test(pos));
  }
  for(; pos <= e; ++pos) {
    EXPECT_TRUE(b.test(pos));
  }
  for(; pos < b.size(); ++pos) {
    EXPECT_FALSE(b.test(pos));
  }
}

TEST(Bitset, RangeConstructor) {
  test_range(0, 10);
  test_range(0, 100);
  test_range(50, 100);
  test_range(10, 10);
  test_range(0, 0);
  test_range(64, 64);
  test_range(63, 63);
  test_range(64, 127);
  test_range(63, 128);
  test_range(0, 63);
  test_range(0, 64);
}

TEST(DynBitset, Assignment) {
  DynBitset b("111111111");
  test_range(0, 8);
  b = DynBitset("1111");
  test_range(0, 3);
}

TEST(DynBitset, Resize) {
  DynBitset b(0, 4);
  EXPECT_EQ(b.size(), CHAR_BIT * sizeof(unsigned long long));
  b.resize((-3) + 100 * CHAR_BIT * sizeof(unsigned long long));
  EXPECT_EQ(b.size(), 100 * CHAR_BIT * sizeof(unsigned long long));
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
  TEST_ALL_STATIC(test_set_and_test);
}

TEST(Bitset, Flip) {
  bitset<19, local_memory, unsigned char> b1("0000001000101010111");
  bitset<19, local_memory, unsigned char> b2("1111110111010101000");
  EXPECT_EQ(b1.count(), b2.size() - b2.count());
  EXPECT_EQ(b1.size() - b1.count(), b2.count());
  EXPECT_NE(b1, b2);
  EXPECT_EQ(~b1, b2);
  EXPECT_EQ(b1, ~b2);
  EXPECT_NE(~b1, ~b2);
  b1.flip();
  EXPECT_EQ(b1, b2);
}

template <class B>
void set_operations() {
  B zero("0000000000000000000");
  B one("0000000000000000001");
  B two("0000000000000000011");
  B b1("1000000011000000000");
  B b2("1000000011000000011");
  B b3("0001000011000010000");
  B b4("0000000011000000000");
  B b5("1001000011000010000");
  B b6("1001000011000010011");
  B b7("1001000000000010000");
  B b8("1001000000000010011");
  B all(zero);
  all.flip();
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

  // Inclusion
  B b9("1111111100011111111");
  B b10("1111111110111111111");

  EXPECT_TRUE(zero.is_subset_of(zero));
  EXPECT_TRUE(one.is_subset_of(one));
  EXPECT_TRUE(b2.is_subset_of(b2));
  EXPECT_TRUE(all.is_subset_of(all));
  EXPECT_TRUE(zero.is_subset_of(one));
  EXPECT_TRUE(zero.is_subset_of(b2));
  EXPECT_TRUE(zero.is_subset_of(all));
  EXPECT_TRUE(one.is_subset_of(two));
  EXPECT_TRUE(b1.is_subset_of(b2));
  EXPECT_TRUE(b1.is_subset_of(b5));
  EXPECT_TRUE(b9.is_subset_of(b10));

  EXPECT_FALSE(one.is_subset_of(zero));
  EXPECT_FALSE(two.is_subset_of(one));
  EXPECT_FALSE(b2.is_subset_of(zero));
  EXPECT_FALSE(all.is_subset_of(zero));
  EXPECT_FALSE(b2.is_subset_of(b1));
  EXPECT_FALSE(b5.is_subset_of(b1));
  EXPECT_FALSE(b10.is_subset_of(b9));

  EXPECT_FALSE(b2.is_subset_of(b3));
  EXPECT_FALSE(b3.is_subset_of(b2));

  EXPECT_FALSE(zero.is_proper_subset_of(zero));
  EXPECT_FALSE(one.is_proper_subset_of(one));
  EXPECT_FALSE(b2.is_proper_subset_of(b2));
  EXPECT_FALSE(all.is_proper_subset_of(all));
  EXPECT_TRUE(zero.is_proper_subset_of(one));
  EXPECT_TRUE(zero.is_proper_subset_of(b2));
  EXPECT_TRUE(zero.is_proper_subset_of(all));
  EXPECT_TRUE(one.is_proper_subset_of(two));
  EXPECT_TRUE(b1.is_proper_subset_of(b2));
  EXPECT_TRUE(b1.is_proper_subset_of(b5));
  EXPECT_TRUE(b9.is_proper_subset_of(b10));

  EXPECT_FALSE(one.is_proper_subset_of(zero));
  EXPECT_FALSE(two.is_proper_subset_of(one));
  EXPECT_FALSE(b2.is_proper_subset_of(zero));
  EXPECT_FALSE(all.is_proper_subset_of(zero));
  EXPECT_FALSE(b2.is_proper_subset_of(b1));
  EXPECT_FALSE(b5.is_proper_subset_of(b1));
  EXPECT_FALSE(b10.is_proper_subset_of(b9));

  EXPECT_FALSE(b2.is_proper_subset_of(b3));
  EXPECT_FALSE(b3.is_proper_subset_of(b2));
}

TEST(Bitset, SetOperations) {
  set_operations<bitset<19, local_memory, unsigned char>>();
  set_operations<dynamic_bitset<local_memory, standard_allocator, unsigned char>>();
}

TEST(Bitset, BitCountingOperations) {
  bitset<19, local_memory, unsigned char> zero;
  bitset<19, local_memory, unsigned char> one("0000000000000000001");
  bitset<19, local_memory, unsigned char> two("0000000000000000011");
  bitset<19, local_memory, unsigned char> b1("1000000011000000000");
  bitset<19, local_memory, unsigned char> b2("1000000011000000011");
  bitset<19, local_memory, unsigned char> b3("1111111100011111111");
  bitset<19, local_memory, unsigned char> b4("1111111110111111111");
  bitset<19, local_memory, unsigned char> all;
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
