#include "test_common.h"
#include "device_test_util.h"

#include "flo/device/vertex_mass.cuh"

TEST(VertexMass, cube)
{
  std::vector<int> h_adjacency {
    0,  6,  7, 10, 11,  // 5 adjacent faces
    0,  1,  7,  8,      // 4 adjacent faces
    0,  1,  2, 11,      // 4 adjacent faces
    1,  2,  3,  8,  9,  // 5 adjacent faces
    2,  3,  4, 10, 11,  // 5 adjacent faces
    3,  4,  5,  9,      // 4 adjacent faces
    4,  5,  6, 10,      // 4 adjacent faces
    5,  6,  7,  8,  9}; // 5 adjacent faces
  std::vector<int> h_valence {5, 4, 4, 5, 5, 4, 4, 5}; 
  // zero offset for first vertex
  std::vector<int> h_cumulative {0,  5,  9, 13, 18, 23, 27, 31, 36}; 
  // cube faces all have area (1*1)/2 = 0.5
  std::vector<flo::real> h_area(12, 0.5);
  // Init the device side arrays
  thrust::device_vector<int> d_adjacency = h_adjacency;
  thrust::device_vector<int> d_valence = h_valence;
  thrust::device_vector<int> d_cumulative_valence = h_cumulative;
  thrust::device_vector<flo::real> d_area = h_area;


  auto d_mass = flo::device::vertex_mass(
      d_area.data(),
      d_adjacency.data(),
      d_valence.data(),
      d_cumulative_valence.data(),
      d_area.size(),
      d_valence.size());

  auto h_mass = device_vector_to_host(d_mass);
	// Test the results
  std::vector<flo::real> expected_mass {
    0.833333, 0.666667, 0.666667, 0.833333, 0.833333, 0.666667, 0.666667, 0.833333};
  using namespace testing;
  EXPECT_THAT(h_mass, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_mass));
}


