#include "test_common.h"
#include "device_test_util.h"
#include <cusp/io/matrix_market.h>
#include "flo/device/vertex_vertex_adjacency.cuh"
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
  DeviceDenseMatrixI d_offsets(6, surf.n_faces());

  // Run the function
  flo::device::adjacency_matrix_offset(
    surf.faces, d_adjacency, d_cumulative_valence, d_offsets);
  HostDenseMatrixI h_offsets = d_offsets;

  auto expected_offsets =
    read_host_dense_matrix<int>(mp + "/adjacency_matrix_offset/offsets.mtx");

  // Test the results
  using namespace testing;
  EXPECT_THAT(h_offsets.values, ElementsAreArray(expected_offsets.values));
}
}  // namespace

#define FLO_ADJACENCY_MATRIX_OFFSET_TEST(NAME) \
  TEST(AdjacencyMatrixOffset, NAME)            \
  {                                            \
    test(#NAME);                               \
  }

FLO_ADJACENCY_MATRIX_OFFSET_TEST(cube)
FLO_ADJACENCY_MATRIX_OFFSET_TEST(spot)
FLO_ADJACENCY_MATRIX_OFFSET_TEST(bunny)

#undef FLO_ADJACENCY_MATRIX_OFFSET_TEST
