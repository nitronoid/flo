#include "test_common.h"
#include "flo/host/flo_matrix_operation.hpp"
#include "flo/host/area.hpp"

#define FACE_AREA_TEST(NAME)                                                 \
  TEST(FaceArea, NAME)                                                       \
  {                                                                          \
    const std::string name = #NAME;                                          \
    const std::string matrix_prefix = "../matrices/" + name;                 \
    const auto& surf =                                                       \
      TestCache::get_mesh<TestCache::HOST>("../models/" + name + ".obj");    \
    auto A = flo::host::area(surf.vertices, surf.faces);                     \
    Eigen::Matrix<flo::real, 1, Eigen::Dynamic> dense_A;                     \
    Eigen::loadMarketVector(dense_A, matrix_prefix + "/area/area.mtx");      \
    gsl::span<flo::real> expected_A{dense_A.data(), (size_t)dense_A.size()}; \
    using namespace testing;                                                 \
    EXPECT_THAT(A, Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_A));    \
  }

FACE_AREA_TEST(cube)
FACE_AREA_TEST(spot)
// FACE_AREA_TEST(dense_sphere_400x400)
// FACE_AREA_TEST(dense_sphere_1000x1000)
// FACE_AREA_TEST(dense_sphere_1500x1500)

#undef FACE_AREA_TEST
