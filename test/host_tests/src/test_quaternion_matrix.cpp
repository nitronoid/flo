#include "test_common.h"
#include "flo/host/flo_matrix_operation.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");

  auto QL = flo::host::to_real_quaternion_matrix(L);

  auto expected_QL = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/quaternion_cotangent_laplacian.mtx");

  EXPECT_MAT_NEAR(QL, expected_QL);
}
}  // namespace

#define FLO_QUATERNION_MATRIX_TEST(NAME) \
  TEST(QuaternionMatrix, NAME)           \
  {                                      \
    test(#NAME);                         \
  }

FLO_QUATERNION_MATRIX_TEST(cube)
FLO_QUATERNION_MATRIX_TEST(spot)
FLO_QUATERNION_MATRIX_TEST(bunny)

#undef FLO_QUATERNION_MATRIX_TEST

