#include "test_common.h"
#include "device_test_util.h"

#include "flo/device/area.cuh"

TEST(FaceArea, cube)
{

  auto cube = make_cube();
  auto raw_vert_ptr = (flo::real3*)(&cube.vertices[0][0]);
  auto raw_face_ptr = (int3*)(&cube.faces[0][0]);

  thrust::device_vector<int3> d_faces(cube.n_faces());
  thrust::copy(raw_face_ptr, raw_face_ptr + cube.n_faces(), d_faces.data());

  thrust::device_vector<flo::real3> d_verts(cube.n_vertices());
  thrust::copy(raw_vert_ptr, raw_vert_ptr + cube.n_vertices(), d_verts.data());


  auto d_area = flo::device::area(d_verts.data(), d_faces.data(), d_faces.size());
  auto h_area = device_vector_to_host(d_area);

	// Test the results
  std::vector<flo::real> expected_area(12, 0.5);
  using namespace testing;
  EXPECT_THAT(h_area, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_area));
}


