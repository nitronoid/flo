#include "test_common.h"
#include "flo/host/divergent_edges.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto X =
    read_dense_matrix<flo::real, 4>(mp + "/similarity_xform/lambda.mtx");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> E;
  flo::host::divergent_edges(surf.vertices, surf.faces, X, L, E);

  auto expected_edges =
    read_dense_matrix<flo::real, 4>(mp + "/divergent_edges/edges.mtx");

  EXPECT_MAT_NEAR(E, expected_edges);
}
}  // namespace

#define FLO_DIVERGENT_EDGES_TEST(NAME) \
  TEST(DivergentEdges, NAME)           \
  {                                    \
    test(#NAME);                       \
  }

FLO_DIVERGENT_EDGES_TEST(cube)
FLO_DIVERGENT_EDGES_TEST(spot)
FLO_DIVERGENT_EDGES_TEST(bunny)

#undef FLO_DIVERGENT_EDGES_TEST

