// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include "shared_ptr.hpp"
#include "allocator.hpp"

using namespace battery;

TEST(SharedPtr, ConstructorAssignment) {
  shared_ptr<int> u1;
  shared_ptr<int> u2(nullptr);
  EXPECT_FALSE(bool(u1));
  EXPECT_FALSE(bool(u2));
  EXPECT_EQ(u1.use_count(), 0);
  EXPECT_EQ(u2.use_count(), 0);
  u1 = make_shared<int, StandardAllocator>(4);
  EXPECT_TRUE(bool(u1));
  EXPECT_EQ(u1.use_count(), 1);
  EXPECT_EQ(*u1, 4);
  u2 = std::move(u1);
  EXPECT_EQ(u1.use_count(), 0);
  EXPECT_EQ(u2.use_count(), 1);
  EXPECT_FALSE(bool(u1));
  EXPECT_EQ(u1.get(), nullptr);
  EXPECT_TRUE(bool(u2));
  EXPECT_EQ(*u2, 4);
  shared_ptr<int> u3(u2);
  EXPECT_EQ(u2.use_count(), 2);
  EXPECT_EQ(u3.use_count(), 2);
  EXPECT_EQ(u2.get(), u3.get());
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

TEST(SharedPtr, ConstructorAssignmentObject) {
  {
    A::n = 0;
    shared_ptr<A> u1;
    shared_ptr<A> u2(nullptr);
    EXPECT_EQ(A::n, 0);
    EXPECT_FALSE(bool(u1));
    EXPECT_FALSE(bool(u2));
    u1 = make_shared<A, StandardAllocator>(4);
    EXPECT_EQ(A::n, 1);
    EXPECT_TRUE(bool(u1));
    EXPECT_EQ(u1->x, 4);
    u2 = std::move(u1);
    EXPECT_EQ(A::n, 1);
    EXPECT_FALSE(bool(u1));
    EXPECT_EQ(u1.get(), nullptr);
    EXPECT_TRUE(bool(u2));
    EXPECT_EQ(u2->x, 4);
    A tmp2(10);
    u1 = make_shared<A, StandardAllocator>(std::move(tmp2));
    EXPECT_EQ(A::n, 2);
    auto u4 = u1;
    EXPECT_EQ(A::n, 2);
    EXPECT_EQ(u4.use_count(), 2);
    EXPECT_EQ(u1.use_count(), 2);
    u1.reset();
    EXPECT_EQ(u4.use_count(), 1);
    EXPECT_EQ(u1.use_count(), 0);
  }
  EXPECT_EQ(A::n, 0);
}