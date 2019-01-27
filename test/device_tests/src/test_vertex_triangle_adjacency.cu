#include "test_common.h"
#include "device_test_util.h"

#include "flo/device/vertex_triangle_adjacency.cuh"

TEST(VertexTriangleAdjacency, cube)
{
  auto cube = make_cube();

  thrust::device_vector<int> d_face_verts(cube.n_faces() * 3);
  thrust::copy_n((&cube.faces[0][0]), cube.n_faces() * 3, d_face_verts.data());

  // Declare device side arrays to dump our results
  thrust::device_vector<int> d_adjacency(cube.n_faces() * 3);
  thrust::device_vector<int> d_valence(cube.n_vertices());
  thrust::device_vector<int> d_cumulative_valence(cube.n_vertices() + 1);

  // Run the function
  flo::device::vertex_triangle_adjacency(
      d_face_verts.data(), 
      cube.n_faces(), 
      cube.n_vertices(), 
      d_adjacency.data(), 
      d_valence.data(), 
      d_cumulative_valence.data());

	// Copy the results back to the host side
  auto h_adjacency = device_vector_to_host(d_adjacency);
  auto h_valence = device_vector_to_host(d_valence);
  auto h_cumulative_valence = device_vector_to_host(d_cumulative_valence);


  std::vector<int> expected_adjacency {
    0,  6,  7, 10, 11,  // 5 adjacent faces
    0,  1,  7,  8,      // 4 adjacent faces
    0,  1,  2, 11,      // 4 adjacent faces
    1,  2,  3,  8,  9,  // 5 adjacent faces
    2,  3,  4, 10, 11,  // 5 adjacent faces
    3,  4,  5,  9,      // 4 adjacent faces
    4,  5,  6, 10,      // 4 adjacent faces
    5,  6,  7,  8,  9}; // 5 adjacent faces
  std::vector<int> expected_valence {5, 4, 4, 5, 5, 4, 4, 5}; 
  // zero offset for first vertex
  std::vector<int> expected_cumulative {0,  5,  9, 13, 18, 23, 27, 31, 36}; 

	// Test the results
  using namespace testing;
  EXPECT_THAT(h_adjacency, ElementsAreArray(expected_adjacency));
  EXPECT_THAT(h_valence, ElementsAreArray(expected_valence));
  EXPECT_THAT(h_cumulative_valence, ElementsAreArray(expected_cumulative));

}

