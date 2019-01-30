#include "test_common.h"

#include "flo/host/vertex_mass.hpp"

TEST(VertexMass, cube)
{
  auto cube = make_cube();

  auto mass = flo::host::vertex_mass(cube.vertices, cube.faces);

  std::vector<flo::real> expected_mass {
    0.833333, 0.666667, 0.666667, 0.833333, 0.833333, 0.666667, 0.666667, 0.833333};
  using namespace testing;
  EXPECT_THAT(mass, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_mass));

}


