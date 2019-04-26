#include "test_common.h"
#include "flo/host/vertex_triangle_adjacency.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  // Declare arrays to dump our results
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTAK;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTA;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTV;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTCV;

  // Run the function
  flo::host::vertex_triangle_adjacency(surf.faces, VTAK, VTA, VTV, VTCV);

  auto expected_VTAK =
    read_vector<int>(mp + "/vertex_triangle_adjacency/adjacency_keys.mtx");
  auto expected_VTA =
    read_vector<int>(mp + "/vertex_triangle_adjacency/adjacency.mtx");
  auto expected_VTV =
    read_vector<int>(mp + "/vertex_triangle_adjacency/valence.mtx");
  auto expected_VTCV =
    read_vector<int>(mp + "/vertex_triangle_adjacency/cumulative_valence.mtx");

  EXPECT_MAT_NEAR(VTAK, expected_VTAK);
  EXPECT_MAT_NEAR(VTA, expected_VTA);
  EXPECT_MAT_NEAR(VTV, expected_VTV);
  EXPECT_MAT_NEAR(VTCV, expected_VTCV);
}
}  // namespace

#define FLO_VERTEX_TRIANGLE_ADJACENCY_TEST(NAME) \
  TEST(VertexTriangleAdjacency, NAME)            \
  {                                              \
    test(#NAME);                                 \
  }

FLO_VERTEX_TRIANGLE_ADJACENCY_TEST(cube)
FLO_VERTEX_TRIANGLE_ADJACENCY_TEST(spot)
FLO_VERTEX_TRIANGLE_ADJACENCY_TEST(bunny)

#undef FLO_VERTEX_TRIANGLE_ADJACENCY_TEST

