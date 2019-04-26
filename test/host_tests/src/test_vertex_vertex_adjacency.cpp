#include "test_common.h"
#include "flo/host/vertex_vertex_adjacency.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  // Declare arrays to dump our results
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVAK;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVA;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVV;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVCV;

  // Run the function
  flo::host::vertex_vertex_adjacency(surf.faces, VVAK, VVA, VVV, VVCV);

  auto expected_VVAK =
    read_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto expected_VVA =
    read_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto expected_VVV =
    read_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto expected_VVCV =
    read_vector<int>(mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");

  EXPECT_MAT_NEAR(VVAK, expected_VVAK);
  EXPECT_MAT_NEAR(VVA, expected_VVA);
  EXPECT_MAT_NEAR(VVV, expected_VVV);
  EXPECT_MAT_NEAR(VVCV, expected_VVCV);
}
}  // namespace

#define FLO_VERTEX_VERTEX_ADJACENCY_TEST(NAME) \
  TEST(VertexVertexAdjacency, NAME)            \
  {                                              \
    test(#NAME);                                 \
  }

FLO_VERTEX_VERTEX_ADJACENCY_TEST(cube)
FLO_VERTEX_VERTEX_ADJACENCY_TEST(spot)
FLO_VERTEX_VERTEX_ADJACENCY_TEST(bunny)

#undef FLO_VERTEX_VERTEX_ADJACENCY_TEST

