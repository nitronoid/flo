#include "test_common.h"
#include "device_test_util.h"

#include "flo/device/area.cuh"

TEST(FaceArea, cube)
{

  auto cube = make_cube();
  thrust::device_vector<Eigen::Vector3i> d_faces = cube.faces;
  thrust::device_vector<Eigen::Vector3d> d_verts = cube.vertices;


  auto d_area = flo::device::area(d_verts.data(), d_faces.data(), d_faces.size());
  auto h_area = device_vector_to_host(d_area);

	// Test the results
  std::vector<double> expected_area(12, 0.5);
  using namespace testing;
  EXPECT_THAT(h_area, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_area));
}


