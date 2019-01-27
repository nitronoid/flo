#ifndef FLO_INCLUDED_TEST_COMMON_MATCHERS
#define FLO_INCLUDED_TEST_COMMON_MATCHERS

#include <gmock/gmock-matchers.h>
#include <string>

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

#endif//FLO_INCLUDED_TEST_COMMON_MATCHERS
