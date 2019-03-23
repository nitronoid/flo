#include "test_common.h"
#include "flo/host/similarity_xform.hpp"

namespace
{
void test(std::string name)
{
  const std::string matrix_prefix = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  Eigen::SparseMatrix<flo::real> D;
  Eigen::loadMarket(D, matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx");

  auto lambda = flo::host::similarity_xform(D);

  Eigen::VectorXf expected_lambda_real;
  Eigen::loadMarketVector(expected_lambda_real,
                          matrix_prefix + "/similarity_xform/lambda.mtx");

  std::vector<Eigen::Matrix<flo::real, 4, 1>> expected_lambda(
    expected_lambda_real.size() / 4);
  std::move(expected_lambda_real.data(),
            expected_lambda_real.data() + expected_lambda_real.size(),
            (float*)&expected_lambda[0][0]);

  using namespace testing;
  EXPECT_THAT(lambda, Pointwise(EigenNear(), expected_lambda));
}
}  // namespace

#define FLO_SIMILARITY_XFORM_TEST(NAME) \
  TEST(SimilarityXform, NAME)           \
  {                                     \
    test(#NAME);                        \
  }

FLO_SIMILARITY_XFORM_TEST(cube)
FLO_SIMILARITY_XFORM_TEST(spot)

#undef FLO_SIMILARITY_XFORM_TEST

