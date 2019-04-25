#include "test_common.h"
#include "flo/host/similarity_xform.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto D =
    read_sparse_matrix<flo::real>(mp + "/intrinsic_dirac/intrinsic_dirac.mtx");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> X;
  flo::host::similarity_xform(D, X);

  auto expected_X =
    read_dense_matrix<flo::real, 4>(mp + "/similarity_xform/lambda.mtx");

  EXPECT_MAT_NEAR(X, expected_X);
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

