#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"

TEST(CotangentLaplacian, cube)
{
  // cube faces all have area (1*1)/2 = 0.5
  std::vector<flo::real> h_area(12, 0.5);
  thrust::device_vector<flo::real> d_area = h_area;

  auto cube = make_cube();
  auto raw_vert_ptr = (flo::real3*)(&cube.vertices[0][0]);
  auto raw_face_ptr = (int3*)(&cube.faces[0][0]);

  thrust::device_vector<int3> d_faces(cube.n_faces());
  thrust::copy(raw_face_ptr, raw_face_ptr + cube.n_faces(), d_faces.data());

  thrust::device_vector<flo::real3> d_verts(cube.n_vertices());
  thrust::copy(raw_vert_ptr, raw_vert_ptr + cube.n_vertices(), d_verts.data());

  int total_valence = 36;

  auto d_L = flo::device::cotangent_laplacian(d_verts.data(),
                                              d_faces.data(),
                                              d_area.data(),
                                              cube.n_vertices(),
                                              cube.n_faces(),
                                              total_valence);

  std::vector<int> I(d_L.row_indices.size());
  std::vector<int> J(d_L.column_indices.size());
  std::vector<flo::real> V(d_L.values.size());

  thrust::copy(d_L.row_indices.begin(), d_L.row_indices.end(), I.begin());
  thrust::copy(d_L.column_indices.begin(), d_L.column_indices.end(), J.begin());
  thrust::copy(d_L.values.begin(), d_L.values.end(), V.begin());

  std::vector<int> expected_I{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                              2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5,
                              5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7};

  std::vector<int> expected_J{0, 1, 2, 4, 6, 7, 0, 1, 2, 3, 7, 0, 1, 2, 3,
                              4, 1, 2, 3, 4, 5, 7, 0, 2, 3, 4, 5, 6, 3, 4,
                              5, 6, 7, 0, 4, 5, 6, 7, 0, 1, 3, 5, 6, 7};

  std::vector<flo::real> expected_V{3,  -1, -1, -0, -1, -0, -1, 3,  -0, -1, -1,
                                    -1, -0, 3,  -1, -1, -1, -1, 3,  -0, -1, -0,
                                    -0, -1, -0, 3,  -1, -1, -1, -1, 3,  -0, -1,
                                    -1, -1, -0, 3,  -1, -0, -1, -0, -1, -1, 3};

  using namespace testing;
  EXPECT_THAT(I, Pointwise(Eq(), expected_I));
  EXPECT_THAT(J, Pointwise(Eq(), expected_J));
  EXPECT_THAT(V, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_V));
}

TEST(CotangentLaplacianOffsets, cube)
{
  // cube faces all have area (1*1)/2 = 0.5
  std::vector<flo::real> h_area(12, 0.5);
  thrust::device_vector<flo::real> d_area = h_area;

  auto cube = make_cube();
  auto raw_vert_ptr = (flo::real3*)(&cube.vertices[0][0]);
  auto raw_face_ptr = (int3*)(&cube.faces[0][0]);

  thrust::device_vector<int3> d_faces(cube.n_faces());
  thrust::copy(raw_face_ptr, raw_face_ptr + cube.n_faces(), d_faces.data());

  thrust::device_vector<flo::real3> d_verts(cube.n_vertices());
  thrust::copy(raw_vert_ptr, raw_vert_ptr + cube.n_vertices(), d_verts.data());

  thrust::device_vector<int> d_valence(cube.n_vertices());
  thrust::device_vector<int> d_cumulative_valence(cube.n_vertices()+1);
  auto d_adjacency =
    flo::device::vertex_vertex_adjacency(d_faces.data(),
                                         cube.n_faces(),
                                         cube.n_vertices(),
                                         d_valence.data(),
                                         d_cumulative_valence.data());

  auto d_offsets =
    flo::device::adjacency_matrix_offset(d_faces.data(),
                                         d_adjacency.data(),
                                         d_cumulative_valence.data(),
                                         cube.n_faces());

  auto d_L = flo::device::cotangent_laplacian(d_verts.data(),
                                              d_faces.data(),
                                              d_area.data(),
                                              d_cumulative_valence.data(),
                                              d_offsets.data(),
                                              cube.n_vertices(),
                                              cube.n_faces(),
                                              36);

  std::vector<int> I(d_L.row_indices.size());
  std::vector<int> J(d_L.column_indices.size());
  std::vector<flo::real> V(d_L.values.size());

  thrust::copy(d_L.row_indices.begin(), d_L.row_indices.end(), I.begin());
  thrust::copy(d_L.column_indices.begin(), d_L.column_indices.end(), J.begin());
  thrust::copy(d_L.values.begin(), d_L.values.end(), V.begin());

  std::vector<int> expected_I{0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 2, 0, 2,
                              2, 3, 3, 0, 3, 3, 3, 4, 4, 4, 0, 4, 4, 5, 5,
                              0, 5, 5, 6, 6, 6, 0, 6, 7, 7, 7, 7, 7, 0};

  std::vector<int> expected_J{0, 1, 2, 4, 6, 7, 0, 0, 2, 3, 7, 0, 1, 0, 3,
                              4, 1, 2, 0, 4, 5, 7, 0, 2, 3, 0, 5, 6, 3, 4,
                              0, 6, 7, 0, 4, 5, 0, 7, 0, 1, 3, 5, 6, 0};

  std::vector<flo::real> expected_V{0,  -1, -1, -0, -1, -0, -1, 0,  -0, -1, -1,
                                    -1, -0, 0,  -1, -1, -1, -1, 0,  -0, -1, -0,
                                    -0, -1, -0, 0,  -1, -1, -1, -1, 0,  -0, -1,
                                    -1, -1, -0, 0,  -1, -0, -1, -0, -1, -1, 0};
  using namespace testing;
  EXPECT_THAT(I, Pointwise(Eq(), expected_I));
  EXPECT_THAT(J, Pointwise(Eq(), expected_J));
  EXPECT_THAT(V, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_V));
}

TEST(CotangentLaplacianOffsets, strip)
{
  std::vector<int> faces{0, 1, 2, 0, 2, 3, 1, 5, 2, 1, 4, 5};

  thrust::device_vector<int3> d_faces(4);
  thrust::copy_n((int3*)(faces.data()), 4, d_faces.data());

  thrust::device_vector<int> d_valence(6);
  thrust::device_vector<int> d_cumulative_valence(7);
  auto d_adjacency = flo::device::vertex_vertex_adjacency(
    d_faces.data(), 4, 6, d_valence.data(), d_cumulative_valence.data());

  auto d_offsets = flo::device::adjacency_matrix_offset(
    d_faces.data(), d_adjacency.data(), d_cumulative_valence.data(), 4);
}

