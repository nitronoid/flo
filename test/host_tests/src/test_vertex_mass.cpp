#include "test_common.h"
#include "flo/host/vertex_mass.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> M;
  flo::host::vertex_mass(surf.vertices, surf.faces, M);

  auto expected_M =
    read_vector<flo::real>(mp + "/vertex_mass/vertex_mass.mtx");

  EXPECT_MAT_NEAR(M, expected_M);
}
}  // namespace

#define FLO_VERTEX_MASS_TEST(NAME) \
  TEST(VertexMass, NAME)           \
  {                                \
    test(#NAME);                   \
  }

FLO_VERTEX_MASS_TEST(cube)
FLO_VERTEX_MASS_TEST(spot)

#undef FLO_VERTEX_MASS_TEST

