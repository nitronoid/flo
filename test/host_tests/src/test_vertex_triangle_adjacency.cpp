#include "test_common.h"
#include "flo/host/vertex_triangle_adjacency.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  // Declare arrays to dump our results
  std::vector<int> adjacency(surf.n_faces() * 3);
  std::vector<int> valence(surf.n_vertices());
  std::vector<int> cumulative_valence(surf.n_vertices() + 1);

  // Run the function
  flo::host::vertex_triangle_adjacency(
    surf.faces, surf.n_vertices(), adjacency, valence, cumulative_valence);

  auto expected_valence =
    read_vector<int>(mp + "/vertex_triangle_adjacency/valence.mtx");
  auto expected_cumulative_valence =
    read_vector<int>(mp + "/vertex_triangle_adjacency/cumulative_valence.mtx");
  auto expected_adjacency =
    read_vector<int>(mp + "/vertex_triangle_adjacency/adjacency.mtx");

  using namespace testing;
  EXPECT_THAT(adjacency, ElementsAreArray(expected_adjacency));
  EXPECT_THAT(valence, ElementsAreArray(expected_valence));
  EXPECT_THAT(cumulative_valence,
              ElementsAreArray(expected_cumulative_valence));
}
}  // namespace

#define FLO_VERTEX_TRIANGLE_ADJACENCY_TEST(NAME) \
  TEST(VertexTriangleAdjacency, NAME)            \
  {                                              \
    test(#NAME);                                 \
  }

FLO_VERTEX_TRIANGLE_ADJACENCY_TEST(cube)
FLO_VERTEX_TRIANGLE_ADJACENCY_TEST(spot)

#undef FLO_VERTEX_TRIANGLE_ADJACENCY_TEST

