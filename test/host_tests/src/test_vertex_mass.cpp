#include "test_common.h"
#include "flo/host/vertex_mass.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto mass = flo::host::vertex_mass(surf.vertices, surf.faces);

  auto expected_mass =
    read_vector<flo::real>(mp + "/vertex_mass/vertex_mass.mtx");

  using namespace testing;
  EXPECT_THAT(mass, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_mass));
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

