#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/area.cuh"
#include <cusp/io/matrix_market.h>

namespace
{
void test(std::string name)
{
  const std::string matrix_prefix = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  cusp::array1d<flo::real, cusp::device_memory> d_area(surf.n_faces());
  flo::device::area(surf.vertices, surf.faces, d_area);
  cusp::array1d<flo::real, cusp::device_memory> h_area = d_area;
  cusp::array1d<flo::real, cusp::host_memory> expected_area;
  cusp::io::read_matrix_market_file(expected_area,
                                    matrix_prefix + "/area/area.mtx");
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
