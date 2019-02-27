#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/area.cuh"

using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;

TEST(CotangentLaplacianTriplets, cube)
{
  const auto& d_cube =
    TestCache::get_mesh<TestCache::DEVICE>("../models/cube.obj");
  thrust::device_vector<flo::real> d_area(12, 0.5f);

  int total_valence = 36;

  auto d_L = flo::device::cotangent_laplacian(d_cube.vertices.data(),
                                              d_cube.faces.data(),
                                              d_area.data(),
                                              d_cube.n_vertices(),
                                              d_cube.n_faces(),
                                              total_valence);
  SparseHostMatrix h_L;
  h_L.column_indices = d_L.column_indices;
  h_L.row_indices = d_L.row_indices;
  h_L.values = d_L.values;

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
  EXPECT_THAT(h_L.row_indices, Pointwise(Eq(), expected_I));
  EXPECT_THAT(h_L.column_indices, Pointwise(Eq(), expected_J));
  EXPECT_THAT(h_L.values, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_V));
}

TEST(CotangentLaplacianAtomic, cube)
{
  // Temp array to read matrix files from disk
  cusp::array1d<int, cusp::host_memory> int_temp;
  cusp::array1d<flo::real, cusp::host_memory> real_temp;

  const auto& d_cube =
    TestCache::get_mesh<TestCache::DEVICE>("../models/cube.obj");

  // Read in all matrices from disk and copy to device memory
  cusp::io::read_matrix_market_file(real_temp, "../matrices/cube/area/area.mtx");
  cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;

  cusp::io::read_matrix_market_file(
    int_temp, "../matrices/cube/vertex_vertex_adjacency/valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_valence = int_temp;

  cusp::io::read_matrix_market_file(
    int_temp, "../matrices/cube/vertex_vertex_adjacency/cumulative_valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;

  cusp::io::read_matrix_market_file(
    int_temp, "../matrices/cube/vertex_vertex_adjacency/adjacency.mtx");
  cusp::array1d<int, cusp::device_memory> d_adjacency = int_temp;

  cusp::array1d<int2, cusp::device_memory> d_offsets(d_cube.n_faces()*3);
  cusp::io::read_matrix_market_file(
    int_temp, "../matrices/cube/adjacency_matrix_offset/offsets.mtx");
  thrust::copy_n((int2*)int_temp.data(),
                 int_temp.size() / 2,
                 d_offsets.data());


  SparseDeviceMatrix d_L(d_cube.n_vertices(),
                         d_cube.n_vertices(),
                         d_cumulative_valence.back() + d_cube.n_vertices());

  cusp::array1d<int, cusp::device_memory> d_diagonals(d_cube.n_vertices());

  flo::device::cotangent_laplacian(d_cube.vertices.data(),
                                   d_cube.faces.data(),
                                   d_area.data(),
                                   d_cumulative_valence.data(),
                                   d_offsets.data(),
                                   d_cube.n_vertices(),
                                   d_cube.n_faces(),
                                   d_cumulative_valence.back(),
                                   d_diagonals.data(),
                                   d_L.row_indices.data(),
                                   d_L.column_indices.data(),
                                   d_L.values.data());

  SparseHostMatrix h_L;
  h_L.column_indices = d_L.column_indices;
  h_L.row_indices = d_L.row_indices;
  h_L.values = d_L.values;
  cusp::array1d<int, cusp::host_memory> h_diagonals = d_diagonals;

  SparseHostMatrix expected_L;
  cusp::io::read_matrix_market_file(
    expected_L, "../matrices/cube/cotangent_laplacian/cotangent_laplacian.mtx");
  cusp::array1d<int, cusp::host_memory> expected_diagonals;
  cusp::io::read_matrix_market_file(
    expected_diagonals, "../matrices/cube/cotangent_laplacian/diagonals.mtx");

  using namespace testing;
  EXPECT_THAT(h_diagonals, Pointwise(Eq(), expected_diagonals));
  EXPECT_THAT(h_L.row_indices, Pointwise(Eq(), expected_L.row_indices));
  EXPECT_THAT(h_L.column_indices, Pointwise(Eq(), expected_L.column_indices));
  EXPECT_THAT(h_L.values,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_L.values));
}

// TEST(WRITE_MAT, mat)
//{
//  auto d_adjacency =
//    flo::device::vertex_vertex_adjacency(d_cube.faces.data(),
//                                         d_cube.n_faces(),
//                                         d_cube.n_vertices(),
//                                         d_valence.data(),
//                                         d_cumulative_valence.data());
//  flo::device::adjacency_matrix_offset(d_cube.faces.data(),
//                                       d_adjacency.data(),
//                                       d_cumulative_valence.data(),
//                                       d_cube.n_faces());
//}
