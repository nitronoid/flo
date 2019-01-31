#include "test_common.h"

#include "flo/host/valence.hpp"

TEST(VertexValence, cube)
{
  auto cube = make_cube();

  auto valence = flo::host::valence(cube.faces);

  using namespace testing;
  // Check that all vertices have valence 8
  EXPECT_THAT(valence, Each(AllOf(Eq(8))));
}


