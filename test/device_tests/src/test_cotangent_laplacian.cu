#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include "flo/device/cotangent_laplacian.cuh"

namespace
{
void test(std::string name)
{
  // Set-up matrix path
  const std::string mp = "../matrices/" + name;
  // Load the surface from our mesh cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read all our dependencies from disk
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_adjacency_keys =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto d_adjacency =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");

  // Allocate a sparse matrix to store our result
  DeviceSparseMatrixR d_L(surf.n_vertices(),
                          surf.n_vertices(),
                          d_cumulative_valence.back() + surf.n_vertices());

  // Run our function
  flo::device::cotangent_laplacian(surf.vertices,
                                   surf.faces,
                                   d_adjacency_keys,
                                   d_adjacency,
                                   d_cumulative_valence,
                                   d_L);

  // Copy our results back to the host side
  HostSparseMatrixR h_L = d_L;

  // Load our expected results from disk
  auto expected_L = read_host_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");

  // test our results
  using namespace testing;
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
FLO_COTANGENT_LAPLACIAN_TEST(bunny)

#undef FLO_COTANGENT_LAPLACIAN_TEST
