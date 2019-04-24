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
  auto lambda =
    read_dense_matrix<flo::real, 4>(mp + "/similarity_xform/lambda.mtx");

  auto edges = flo::host::divergent_edges(surf.vertices, surf.faces, lambda, L);

  auto expected_edges =
    read_dense_matrix<flo::real, 4>(mp + "/divergent_edges/edges.mtx");

  using namespace testing;
  EXPECT_THAT(expected_edges, Pointwise(EigenNear(), edges));
}
}  // namespace

#define FLO_DIVERGENT_EDGES_TEST(NAME) \
  TEST(DivergentEdges, NAME)           \
  {                                    \
    test(#NAME);                       \
  }

FLO_DIVERGENT_EDGES_TEST(cube)
FLO_DIVERGENT_EDGES_TEST(spot)

#undef FLO_DIVERGENT_EDGES_TEST

