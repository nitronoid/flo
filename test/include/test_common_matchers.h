#ifndef FLO_INCLUDED_TEST_COMMON_MATCHERS
#define FLO_INCLUDED_TEST_COMMON_MATCHERS

#include <gmock/gmock-matchers.h>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

MATCHER_P(EigenNearP,
          expect,
          "")
{
  return arg.isApprox((const decltype(arg)&)expect, FLOAT_SOFT_EPSILON);
}

#define EXPECT_MAT_NEAR(A, B) \
  EXPECT_THAT(A, EigenNearP(::testing::ByRef(B)))

MATCHER(EigenNear,"")
{
  return ::testing::get<0>(arg).isApprox(::testing::get<1>(arg), FLOAT_SOFT_EPSILON);
}

namespace Eigen
{
template <typename Real, int R, int C>
void PrintTo(const Eigen::Matrix<Real, R, C> &m, std::ostream *os)
{
  *os << '\n' << m;
}
template void PrintTo(const Eigen::Matrix<double, 2, 1>& m, std::ostream *os);
template void PrintTo(const Eigen::Matrix<double, 3, 1>& m, std::ostream *os);
template void PrintTo(const Eigen::Matrix<double, 4, 1>& m, std::ostream *os);
template void PrintTo(const Eigen::Matrix<double, Dynamic, 1>& m, std::ostream *os);
template void PrintTo(const Eigen::Matrix<double, Dynamic, Dynamic>& m, std::ostream *os);
//inline void PrintTo(const Eigen::MatrixXd &m, std::ostream *os)
//{
//  *os << '\n' << m;
//}
}

#endif//FLO_INCLUDED_TEST_COMMON_MATCHERS
