#include "test_common.h"

#include "flo/host/vertex_triangle_adjacency.hpp"

TEST(VertexTriangleAdjacency, cube)
{
  auto cube = make_cube();

  // Declare arrays to dump our results
  std::vector<int> adjacency(cube.n_faces() * 3);
  std::vector<int> valence(cube.n_vertices());
  std::vector<int> cumulative_valence(cube.n_vertices() + 1);

  // Run the function
  flo::host::vertex_triangle_adjacency(
      cube.faces, cube.n_vertices(), adjacency, valence, cumulative_valence);

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

  using namespace testing;
  EXPECT_THAT(adjacency, ElementsAreArray(expected_adjacency));
  EXPECT_THAT(valence, ElementsAreArray(expected_valence));
  EXPECT_THAT(cumulative_valence, ElementsAreArray(expected_cumulative));

}

