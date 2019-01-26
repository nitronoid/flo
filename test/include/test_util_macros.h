#ifndef INCLUDED_TESTUTILMACROS_H
#define INCLUDED_TESTUTILMACROS_H

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>

#define FLOAT_SOFT_EPSILON 0.005f
#define EXPECT_FLOAT_NEAR(A, B) EXPECT_NEAR(A, B, FLOAT_SOFT_EPSILON);

#define EXPECT_VEC3_EQ(A, B)       \
  EXPECT_FLOAT_EQ((A)[0], (B)[0]); \
  EXPECT_FLOAT_EQ((A)[1], (B)[1]); \
  EXPECT_FLOAT_EQ((A)[2], (B)[2]);

#define EXPECT_VEC3_NE(A, B) \
  EXPECT_NE((A)[0], (B)[0]); \
  EXPECT_NE((A)[1], (B)[1]); \
  EXPECT_NE((A)[2], (B)[2]);

#define EXPECT_VEC2_EQ(A, B)       \
  EXPECT_FLOAT_EQ((A)[0], (B)[0]); \
  EXPECT_FLOAT_EQ((A)[1], (B)[1]);

#define EXPECT_VEC2_NE(A, B) \
  EXPECT_NE((A)[0], (B)[0]); \
  EXPECT_NE((A)[1], (B)[1]);

#define EXPECT_VEC3_NEAR(A, B)       \
  EXPECT_FLOAT_NEAR((A)[0], (B)[0]); \
  EXPECT_FLOAT_NEAR((A)[1], (B)[1]); \
  EXPECT_FLOAT_NEAR((A)[2], (B)[2]);

#define EXPECT_VEC2_NEAR(A, B)       \
  EXPECT_FLOAT_NEAR((A)[0], (B)[0]); \
  EXPECT_FLOAT_NEAR((A)[1], (B)[1]);

MATCHER_P(EigenNear,
          expect,
          std::string(negation ? "isn't" : "is") + " near to" +
            ::testing::PrintToString(expect))
{
  return arg.isApprox(expect, FLOAT_SOFT_EPSILON);
}

template <class Base>
class EigenPrintWrap : public Base
{
  friend void PrintTo(const EigenPrintWrap& m, ::std::ostream* o)
  {
    *o << "\n" << m;
  }
};

template <class Base>
const EigenPrintWrap<Base>& print_wrap(const Base& base)
{
  return static_cast<const EigenPrintWrap<Base>&>(base);
}

#define EXPECT_MAT_NEAR(A, B) \
  EXPECT_THAT(print_wrap(A), EigenNear(print_wrap(B)));

#endif  // INCLUDED_TESTUTILMACROS_H
