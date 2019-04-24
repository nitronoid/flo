#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/vertex_mass.cuh"

#include "flo/host/cotangent_laplacian.hpp"

namespace
{
void test(std::string name)
{
  // Set-up matrix path
  const std::string mp = "../matrices/" + name;
  // Load the surface from our mesh cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read all our dependencies from disk
  auto d_valence =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_adjacency_keys =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto d_adjacency =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto d_offsets =
    read_device_dense_matrix<int>(mp + "/adjacency_matrix_offset/offsets.mtx");

  // Allocate a sparse matrix to store our result
  DeviceSparseMatrixR d_L(surf.n_vertices(),
                          surf.n_vertices(),
                          d_cumulative_valence.back() + surf.n_vertices());

  // Allocate a dense 1 dimensional array to receive diagonal element indices
  DeviceVectorI d_diagonals(surf.n_vertices());
  // Run our function
  flo::device::cotangent_laplacian(surf.vertices,
                                   surf.faces,
                                   d_offsets,
                                   d_adjacency_keys,
                                   d_adjacency,
                                   d_cumulative_valence,
                                   d_diagonals,
                                   d_L);

  // Copy our results back to the host side
  HostSparseMatrixR h_L = d_L;
  HostVectorI h_diagonals = d_diagonals;

  // Load our expected results from disk
  auto expected_L = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto expected_diagonals =
    read_device_vector<int>(mp + "/cotangent_laplacian/diagonals.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_diagonals, Pointwise(Eq(), expected_diagonals));
  EXPECT_THAT(h_L.row_indices, Pointwise(Eq(), expected_L.row_indices));
  EXPECT_THAT(h_L.column_indices, Pointwise(Eq(), expected_L.column_indices));
  EXPECT_THAT(h_L.values,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_L.values));
}
}  // namespace

#define FLO_COTANGENT_LAPLACIAN_TEST(NAME) \
  TEST(CotangentLaplacian, NAME)           \
  {                                        \
    test(#NAME);                           \
  }

FLO_COTANGENT_LAPLACIAN_TEST(cube)
FLO_COTANGENT_LAPLACIAN_TEST(spot)

#undef FLO_COTANGENT_LAPLACIAN_TEST
