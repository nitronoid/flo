#include "test_common.h"

#include "flo/host/area.hpp"

TEST(FaceArea, cube)
{
  auto cube = make_cube();

  auto face_area = flo::host::area(cube.vertices, cube.faces);

  using namespace testing;
#ifdef FLO_USE_DOUBLE_PRECISION
  EXPECT_THAT(face_area, Each(AllOf(DoubleNear(0.5, FLOAT_SOFT_EPSILON))));
#else
  EXPECT_THAT(face_area, Each(AllOf(FloatNear(0.5, FLOAT_SOFT_EPSILON))));
#endif
  // std::vector<flo::real> expected_area(12, 0.5f);
  // EXPECT_THAT(face_area, Pointwise(FloatNear(FLOAT_SOFT_EPSILON),
  // expected_area));
}

