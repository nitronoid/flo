#include "test_common.h"

#include "flo/host/area.hpp"

TEST(FaceArea, cube)
{
  auto cube = make_cube();

  auto face_area = flo::host::area(cube.vertices, cube.faces);

  std::vector<double> expected_area(12, 0.5);
  using namespace testing;
  EXPECT_THAT(face_area, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_area));
}


