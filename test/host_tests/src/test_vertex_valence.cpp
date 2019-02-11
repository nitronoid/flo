#include "test_common.h"

#include "flo/host/valence.hpp"

TEST(VertexValence, cube)
{
  auto cube = make_cube();

  auto valence = flo::host::valence(cube.faces);

  int expected_valence[] = {5, 4, 4, 5, 5, 4, 4, 5};
  
  using namespace testing;
  // Check that all vertices have valence 8
  EXPECT_THAT(valence, Pointwise(Eq(), expected_valence));
  //EXPECT_THAT(valence, Each(AllOf(Eq(8))));
}


