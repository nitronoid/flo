#include "test_common.h"
#include "flo/host/valence.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto valence = flo::host::valence(surf.faces);

  auto expected_valence =
    read_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");

  using namespace testing;
  EXPECT_THAT(valence, Pointwise(Eq(), expected_valence));
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

