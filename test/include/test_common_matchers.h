#ifndef FLO_INCLUDED_TEST_COMMON_MATCHERS
#define FLO_INCLUDED_TEST_COMMON_MATCHERS

#include <gmock/gmock-matchers.h>
#include <string>
#include <Eigen/Dense>

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

MATCHER(EigenNear2,"")
{
  return ::testing::get<0>(arg).isApprox(::testing::get<1>(arg), FLOAT_SOFT_EPSILON);
}

namespace Eigen
{
inline void PrintTo(const Eigen::Vector4d &m, std::ostream *os)
{
  *os << '\n' << m.transpose();
}
inline void PrintTo(const Eigen::Vector4f &m, std::ostream *os)
{
  *os << '\n' << m.transpose();
}
inline void PrintTo(const Eigen::Vector3d &m, std::ostream *os)
{
  *os << '\n' << m.transpose();
}
inline void PrintTo(const Eigen::Vector3f &m, std::ostream *os)
{
  *os << '\n' << m.transpose();
}
}

#endif//FLO_INCLUDED_TEST_COMMON_MATCHERS
