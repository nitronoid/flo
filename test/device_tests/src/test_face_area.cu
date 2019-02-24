#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/area.cuh"

TEST(FaceArea, cube)
{
  const auto& d_cube =
    TestCache::get_mesh<TestCache::DEVICE>("../models/cube.obj");

  auto d_area = flo::device::area(
    d_cube.vertices.data(), d_cube.faces.data(), d_cube.n_faces());
  auto h_area = device_vector_to_host(d_area);

  // Test the results
  std::vector<flo::real> expected_area(12, 0.5);
  using namespace testing;
  EXPECT_THAT(h_area, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_area));
}

