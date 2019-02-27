#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/area.cuh"
#include <cusp/io/matrix_market.h>

TEST(FaceArea, cube)
{
  const auto& d_cube =
    TestCache::get_mesh<TestCache::DEVICE>("../models/cube.obj");

  auto d_area = flo::device::area(
    d_cube.vertices.data(), d_cube.faces.data(), d_cube.n_faces());
  auto h_area = device_vector_to_host(d_area);

  // Test the results
  cusp::array1d<flo::real, cusp::host_memory> expected_area;
  cusp::io::read_matrix_market_file(expected_area,
                                    "../matrices/cube/area/area.mtx");
  using namespace testing;
  EXPECT_THAT(h_area, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_area));
}

