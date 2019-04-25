#include "test_common.h"
#include <igl/doublearea.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> A;
  igl::doublearea(surf.vertices, surf.faces, A);
  A *= 0.5f;
  auto expected_A = read_vector<flo::real>(mp + "/face_area/face_area.mtx");

  EXPECT_MAT_NEAR(A, expected_A);
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
