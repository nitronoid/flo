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

  auto lambda = flo::host::similarity_xform(D);

  auto expected_lambda =
    read_dense_matrix<flo::real, 4>(mp + "/similarity_xform/lambda.mtx");

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

