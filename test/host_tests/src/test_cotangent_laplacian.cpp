#include "test_common.h"
#include <igl/cotmatrix.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  Eigen::SparseMatrix<flo::real> L;
  igl::cotmatrix(surf.vertices, surf.faces, L);
  L = -(L.eval());

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
FLO_COTANGENT_LAPLACIAN_TEST(bunny)

#undef FLO_COTANGENT_LAPLACIAN_TEST

