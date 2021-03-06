// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "darray.hpp"
#include "allocator.hpp"

using namespace battery;

template<typename Allocator>
void test_array(const DArray<int, Allocator>& a, size_t size, int elem) {
  EXPECT_EQ(a.size(), size);
  for(int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], elem);
  }
}

DArray<int, StandardAllocator>& get_ref(DArray<int, StandardAllocator>& x) { return x; }

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
  DArray<int, StandardAllocator> a6;
  test_array(a6, 0, 0);
  DArray<int, StandardAllocator> a7(std::move(a1));
  test_array(a7, 3, 2);
  DArray<int, StandardAllocator> a8 = a7;
  test_array(a8, 3, 2);
  DArray<int, StandardAllocator> a9;
  get_ref(a9) = std::move(a8);
  test_array(a9, 3, 2);
}

TEST(DArray, Equality) { // NOTE: Further (indirectly) tested in string_test.
  DArray<int, StandardAllocator> a1({1,2,3});
  DArray<int, StandardAllocator> a2({1,2,3});
  DArray<int, StandardAllocator> a3({2,3,4});
  EXPECT_TRUE(a1 == a2);
  EXPECT_TRUE(a2 == a1);
  EXPECT_FALSE(a1 == a3);
  EXPECT_FALSE(a3 == a1);
}

class A {
public:
  int uid;
  A(int i): uid(i){}
  virtual int id() { return uid; }
  virtual A* clone(StandardAllocator& alloc) const {
    return new(alloc) A(uid);
  }
};

class B: public A {
public:
  B(int i): A(i) {}
  virtual int id() { return uid * 10; }
  virtual B* clone(StandardAllocator& alloc) const {
    return new(alloc) B(uid);
  }
};


TEST(DArray, PolymorphicStorage) {
  StandardAllocator standard_allocator;
  DArray<A*, StandardAllocator> poly(3);
  poly[0] = new(standard_allocator) A(1);
  poly[1] = new(standard_allocator) B(2);
  poly[2] = new(standard_allocator) B(3);
  int uids[3] = {1,20,30};
  for(int i = 0; i < poly.size(); ++i) {
    EXPECT_EQ(poly[i]->id(), uids[i]);
  }

  DArray<A*, StandardAllocator> poly2(poly);
  EXPECT_EQ(poly2.size(), poly.size());
  poly2[0]->uid = 5;
  EXPECT_EQ(poly[0]->id(), uids[0]);
  for(int i = 0; i < poly.size(); ++i) {
    EXPECT_NE(poly[i], poly2[i]);
  }

  std::vector<A*> poly3({new A(2), new B(4), new B(5)});
  int uids3[3] = {2, 40, 50};
  for(int i = 0; i < 3; ++i) {
    EXPECT_EQ(poly3[i]->id(), uids3[i]);
  }

  DArray<A*, StandardAllocator> poly4(poly3);
  EXPECT_EQ(poly4.size(), poly3.size());
  for(int i = 0; i < poly4.size(); ++i) {
    EXPECT_EQ(poly4[i]->id(), uids3[i]);
    EXPECT_NE(poly3[i], poly4[i]);
  }

  B* b = new B(2);
  DArray<A*, StandardAllocator> poly5(3, b);
  EXPECT_EQ(poly5.size(), 3);
  for(int i = 0; i < poly5.size(); ++i) {
    EXPECT_EQ(poly5[i]->id(), 20);
    EXPECT_NE(poly5[i], b);
    for(int j = i + 1; j < poly5.size(); ++j) {
      EXPECT_NE(poly5[i], poly5[j]);
    }
  }

  for(auto x : poly3) {
    delete x;
  }
  delete b;
}
