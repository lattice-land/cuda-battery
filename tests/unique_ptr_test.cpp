// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "unique_ptr.hpp"
#include "allocator.hpp"

using namespace battery;

TEST(UniquePtr, ConstructorAssignment) {
  unique_ptr<int> u1;
  unique_ptr<int> u2(nullptr);
  EXPECT_FALSE(bool(u1));
  EXPECT_FALSE(bool(u2));
  u1 = make_unique<int, StandardAllocator>(4);
  EXPECT_TRUE(bool(u1));
  EXPECT_EQ(*u1, 4);
  u2 = std::move(u1);
  EXPECT_FALSE(bool(u1));
  EXPECT_EQ(u1.get(), nullptr);
  EXPECT_TRUE(bool(u2));
  EXPECT_EQ(*u2, 4);
  int* i = u2.release();
  delete i;
}

class A {
  public:
    static int n;
    int x;
    A(): x(0) { ++n; }
    A(A&& a): x(a.x) { a.x = -1; }
    A(const A& a): x(a.x) { ++n; }
    A(int x): x(x) { ++n; }
    ~A() { if(x != -1) --n; }
};

int A::n = 0;

TEST(UniquePtr, ConstructorAssignmentObject) {
  {
    A::n = 0;
    unique_ptr<A> u1;
    unique_ptr<A> u2(nullptr);
    EXPECT_EQ(A::n, 0);
    EXPECT_FALSE(bool(u1));
    EXPECT_FALSE(bool(u2));
    A tmp(4);
    u1 = make_unique<A, StandardAllocator>(tmp);
    EXPECT_EQ(A::n, 2);
    EXPECT_TRUE(bool(u1));
    EXPECT_EQ(u1->x, 4);
    u2 = std::move(u1);
    EXPECT_EQ(A::n, 2);
    EXPECT_FALSE(bool(u1));
    EXPECT_EQ(u1.get(), nullptr);
    EXPECT_TRUE(bool(u2));
    EXPECT_EQ(u2->x, 4);
    A tmp2(10);
    u1 = make_unique<A, StandardAllocator>(std::move(tmp2));
    EXPECT_EQ(A::n, 3);
    A* i = u2.release();
    delete i;
    EXPECT_EQ(A::n, 2);
  }
  EXPECT_EQ(A::n, 0);
}

class B: public A {};

TEST(UniquePtr, TestInheritanceInit) {
  A::n = 0;
  unique_ptr<B> b(new B);
  unique_ptr<A> a = std::move(b);
  EXPECT_EQ(A::n, 1);
}
