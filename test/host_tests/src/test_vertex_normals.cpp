#include "test_common.h"
#include <igl/per_vertex_normals.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> N;
  igl::per_vertex_normals(
    surf.vertices, surf.faces, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);

  auto expected_N =
    read_dense_matrix<flo::real, 3>(mp + "/vertex_normals/vertex_normals.mtx");

  EXPECT_MAT_NEAR(N, expected_N);
}
}  // namespace

#define FLO_VERTEX_NORMALS_TEST(NAME) \
  TEST(VertexNormals, NAME)           \
  {                                   \
    test(#NAME);                      \
  }

FLO_VERTEX_NORMALS_TEST(cube)
FLO_VERTEX_NORMALS_TEST(spot)
FLO_VERTEX_NORMALS_TEST(bunny)

#undef FLO_VERTEX_NORMALS_TEST
