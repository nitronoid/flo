#include "test_common.h"
#include "device_test_util.h"
#include <cusp/io/matrix_market.h>
#include "flo/device/adjacency_matrix_indices.cuh"
#include <cusp/print.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read input matrices from disk
  auto d_adjacency_keys =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto d_adjacency =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto d_valence =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");

  // Declare device side arrays to dump our results
  DeviceDenseMatrixI d_indices(6, surf.n_faces());

  // Run the function
  flo::device::adjacency_matrix_indices(
    surf.faces, d_adjacency, d_cumulative_valence, d_indices);
  HostDenseMatrixI h_indices = d_indices;

  auto expected_indices =
    read_host_dense_matrix<int>(mp + "/adjacency_matrix_indices/indices.mtx");

  // Test the results
  using namespace testing;
  EXPECT_THAT(h_indices.values, ElementsAreArray(expected_indices.values));
}
}  // namespace

#define FLO_ADJACENCY_MATRIX_INDICES_TEST(NAME) \
  TEST(AdjacencyMatrixIndices, NAME)            \
  {                                             \
    test(#NAME);                                \
  }

FLO_ADJACENCY_MATRIX_INDICES_TEST(cube)
FLO_ADJACENCY_MATRIX_INDICES_TEST(spot)
FLO_ADJACENCY_MATRIX_INDICES_TEST(bunny)

#undef FLO_ADJACENCY_MATRIX_INDICES_TEST
