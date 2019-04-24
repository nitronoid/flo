#include "test_common.h"
#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/area.hpp"
#include "flo/host/valence.hpp"
#include "flo/host/flo_matrix_operation.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto L = flo::host::cotangent_laplacian(surf.vertices, surf.faces);

  auto expected_L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");

  EXPECT_MAT_NEAR(L, expected_L);
}
}  // namespace

#define FLO_COTANGENT_LAPLACIAN_TEST(NAME) \
  TEST(CotangentLaplacian, NAME)           \
  {                                        \
    test(#NAME);                           \
  }

FLO_COTANGENT_LAPLACIAN_TEST(cube)
FLO_COTANGENT_LAPLACIAN_TEST(spot)

#undef FLO_COTANGENT_LAPLACIAN_TEST

