#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/face_area.cuh"
#include <cusp/io/matrix_market.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  DeviceVectorR d_area(surf.n_faces());
  flo::device::face_area(surf.vertices, surf.faces, d_area);
  HostVectorR h_area = d_area;

  auto expected_area =
    read_host_vector<flo::real>(mp + "/face_area/face_area.mtx");

  using namespace testing;
  EXPECT_THAT(h_area, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_area));
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
