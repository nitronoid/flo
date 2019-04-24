#include "test_common.h"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/area.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto A = flo::host::area(surf.vertices, surf.faces);

  auto expected_A = read_vector<flo::real>(mp + "/face_area/face_area.mtx");

  using namespace testing;
  EXPECT_THAT(A, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_A));
}
}  // namespace

#define FLO_FACE_AREA_TEST(NAME) \
  TEST(FaceArea, NAME)           \
  {                              \
    test(#NAME);                 \
  }

FLO_FACE_AREA_TEST(cube)
FLO_FACE_AREA_TEST(spot)

#undef FLO_FACE_AREA_TEST
