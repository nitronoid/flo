#include "test_common.h"
#include "device_test_util.h"
#include <cusp/io/matrix_market.h>
#include "flo/device/vertex_vertex_adjacency.cuh"

namespace
{
void test(std::string name)
{
  const std::string matrix_prefix = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Declare device side arrays to dump our results
  cusp::array1d<int, cusp::device_memory> d_adjacency(surf.n_faces() * 12);
  cusp::array1d<int, cusp::device_memory> d_valence(surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
    surf.n_vertices() + 1);

  // Run the function
  int n_adjacency = flo::device::vertex_vertex_adjacency(
    surf.faces,
    d_adjacency,
    d_valence,
    {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()});

  // Copy the results back to the host side
  cusp::array1d<int, cusp::host_memory> h_adjacency =
    d_adjacency.subarray(0, n_adjacency);
  cusp::array1d<int, cusp::host_memory> h_valence = d_valence;
  cusp::array1d<int, cusp::host_memory> h_cumulative_valence =
    d_cumulative_valence;

  // Read expected results from disk
  cusp::array1d<int, cusp::host_memory> expected_valence;
  cusp::io::read_matrix_market_file(
    expected_valence, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");

  cusp::array1d<int, cusp::host_memory> expected_cumulative_valence;
  cusp::io::read_matrix_market_file(
    expected_cumulative_valence,
    matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");

  cusp::array1d<int, cusp::host_memory> expected_adjacency;
  cusp::io::read_matrix_market_file(expected_adjacency,
                                    matrix_prefix +
                                      "/vertex_vertex_adjacency/adjacency.mtx");

  // Test the results
  using namespace testing;
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

#undef FLO_VERTEX_VERTEX_ADJACENCY_TEST
