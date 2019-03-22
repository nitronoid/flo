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

  // Read input matrices from disk
  cusp::array1d<int, cusp::host_memory> int_temp;
  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_valence = int_temp;

  cusp::io::read_matrix_market_file(
    int_temp,
    matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;

  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/vertex_vertex_adjacency/adjacency.mtx");
  cusp::array1d<int, cusp::device_memory> d_adjacency = int_temp;

  // Declare device side arrays to dump our results
  cusp::array1d<int2, cusp::device_memory> d_offsets(surf.n_faces() * 3);

  // Run the function
  flo::device::adjacency_matrix_offset(
    surf.faces, d_adjacency, d_cumulative_valence, d_offsets);

  // Copy the results back to the host side
  cusp::array1d<int2, cusp::host_memory> h_offset_pairs = d_offsets;
  cusp::array1d<int, cusp::host_memory>::const_view h_offsets{
    reinterpret_cast<const int*>(h_offset_pairs.begin().base()),
    reinterpret_cast<const int*>(h_offset_pairs.end().base())};

  cusp::array1d<int, cusp::host_memory> expected_offsets;
  cusp::io::read_matrix_market_file(
    expected_offsets, matrix_prefix + "/adjacency_matrix_offset/offsets.mtx");

  // Test the results
  using namespace testing;
  EXPECT_THAT(h_offsets, ElementsAreArray(expected_offsets));
}
}  // namespace

#define FLO_ADJACENCY_MATRIX_OFFSET_TEST(NAME) \
  TEST(AdjacencyMatrixOffset, NAME)            \
  {                                            \
    test(#NAME);                               \
  }

FLO_ADJACENCY_MATRIX_OFFSET_TEST(cube)
FLO_ADJACENCY_MATRIX_OFFSET_TEST(spot)

#undef FLO_ADJACENCY_MATRIX_OFFSET_TEST
