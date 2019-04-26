#include "test_common.h"
#include "device_test_util.h"
#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include "flo/device/vertex_vertex_adjacency.cuh"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Declare device side arrays to dump our results
  DeviceVectorI d_adjacency(surf.n_faces() * 6);
  DeviceVectorI d_adjacency_keys(surf.n_faces() * 6);
  DeviceVectorI d_valence(surf.n_vertices());
  DeviceVectorI d_cumulative_valence(surf.n_vertices() + 1);

  // Run the function
  int n_adjacency = flo::device::vertex_vertex_adjacency(
    surf.faces,
    d_adjacency_keys,
    d_adjacency,
    d_valence,
    d_cumulative_valence.subarray(1, surf.n_vertices()));

  // Copy the results back to the host side
  HostVectorI h_adjacency_keys = d_adjacency_keys.subarray(0, n_adjacency);
  HostVectorI h_adjacency = d_adjacency.subarray(0, n_adjacency);
  HostVectorI h_valence = d_valence;
  HostVectorI h_cumulative_valence = d_cumulative_valence;

  // Read expected results from disk
  auto expected_adjacency_keys =
    read_host_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto expected_adjacency =
    read_host_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto expected_valence =
    read_host_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto expected_cumulative_valence = read_host_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");

  // Test the results
  using namespace testing;
  EXPECT_THAT(h_adjacency_keys, ElementsAreArray(expected_adjacency_keys));
  EXPECT_THAT(h_adjacency, ElementsAreArray(expected_adjacency));
  EXPECT_THAT(h_valence, ElementsAreArray(expected_valence));
  EXPECT_THAT(h_cumulative_valence,
              ElementsAreArray(expected_cumulative_valence));
}
}  // namespace

#define FLO_VERTEX_VERTEX_ADJACENCY_TEST(NAME) \
  TEST(VertexVertexAdjacency, NAME)            \
  {                                            \
    test(#NAME);                               \
  }

FLO_VERTEX_VERTEX_ADJACENCY_TEST(cube)
FLO_VERTEX_VERTEX_ADJACENCY_TEST(spot)
FLO_VERTEX_VERTEX_ADJACENCY_TEST(bunny)

#undef FLO_VERTEX_VERTEX_ADJACENCY_TEST
