#include "test_common.h"

#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/area.hpp"
#include "flo/host/valence.hpp"

// TEST(IntrinsicDirac, cube)
//{
//  auto cube = make_cube();
//
//  std::vector<int> valence(8, 8);
//  std::vector<flo::real> face_area(12, 0.5f);
//  std::vector<flo::real> rho(8, 3.0f);
//
//  auto D = flo::host::intrinsic_dirac(
//    cube.vertices, cube.faces, valence, face_area, rho);
//
//  Eigen::Matrix<flo::real, 32, 32> dense_D(32, 32);
//  const flo::real o = 1.0;
//  const flo::real z = 0.0;
//  const flo::real f = 5.0;
//  const flo::real m = 5.5;
//  const flo::real h = 1.5;
//  dense_D << m, -z, -z, -z, -z, -o, h, h, -z, -h, o, -h, z, z, z, z, o, -z,
//  -o,
//    -o, z, z, z, z, -z, h, -h, -z, o, o, -z, o, z, m, -z, z, o, -z, h, -h, h,
//    -z, -h, -o, z, z, z, z, z, o, -o, o, z, z, z, z, -h, -z, -z, h, -o, o, o,
//    z, z, z, m, -z, -h, -h, -z, -o, -o, h, -z, -h, z, z, z, z, o, o, o, -z, z,
//    z, z, z, h, z, -z, h, z, -o, o, o, z, -z, z, m, -h, h, o, -z, h, o, h, -z,
//    z, z, z, z, o, -o, z, o, z, z, z, z, z, -h, -h, -z, -o, -z, -o, o, -z, o,
//    -h, -h, f, -z, -z, -z, o, -o, -o, -z, -z, -h, o, h, z, z, z, z, z, z, z,
//    z, z, z, z, z, -z, h, h, -z, -o, -z, -h, h, z, f, -z, z, o, o, -z, o, h,
//    -z, h, -o, z, z, z, z, z, z, z, z, z, z, z, z, -h, -z, -z, -h, h, h, -z,
//    o, z, z, f, -z, o, z, o, -o, -o, -h, -z, -h, z, z, z, z, z, z, z, z, z, z,
//    z, z, -h, z, -z, h, h, -h, -o, -z, z, -z, z, f, z, -o, o, o, -h, o, h, -z,
//    z, z, z, z, z, z, z, z, z, z, z, z, z, h, -h, -z, -z, h, -o, h, o, o, o,
//    -z, f, -z, -z, -z, -z, -o, h, -h, -z, -h, -h, -z, z, z, z, z, z, z, z, z,
//    z, z, z, z, -h, -z, h, o, -o, o, -z, -o, z, f, -z, z, o, -z, -h, -h, h,
//    -z, -z, h, z, z, z, z, z, z, z, z, z, z, z, z, o, -h, -z, h, -o, z, o, o,
//    z, z, f, -z, -h, h, -z, -o, h, z, -z, -h, z, z, z, z, z, z, z, z, z, z, z,
//    z, -h, -o, -h, -z, z, o, -o, o, z, -z, z, f, h, h, o, -z, z, -h, h, -z, z,
//    z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, -z, h, -o, -h, -z, o, -h, h,
//    m, -z, -z, -z, o, -o, -z, o, -z, -h, h, -z, z, z, z, z, o, -z, o, -o, z,
//    z, z, z, -h, -z, -h, o, -o, -z, h, h, z, m, -z, z, o, o, o, z, h, -z, -z,
//    -h, z, z, z, z, z, o, -o, -o, z, z, z, z, o, h, -z, h, h, -h, -z, o, z, z,
//    m, -z, z, -o, o, -o, -h, z, -z, -h, z, z, z, z, -o, o, o, -z, z, z, z, z,
//    h, -o, -h, -z, -h, -h, -o, -z, z, -z, z, m, -o, -z, o, o, z, h, h, -z, z,
//    z, z, z, o, o, z, o, o, -z, o, o, z, z, z, z, -z, h, h, -z, o, o, -z, -o,
//    m, -z, -z, -z, -z, -o, -h, -h, -z, -h, -o, h, z, z, z, z, z, o, o, -o, z,
//    z, z, z, -h, -z, -z, -h, -o, o, -o, z, z, m, -z, z, o, -z, -h, h, h, -z,
//    h, o, z, z, z, z, -o, -o, o, -z, z, z, z, z, -h, z, -z, h, z, o, o, o, z,
//    z, m, -z, h, h, -z, -o, o, -h, -z, -h, z, z, z, z, -o, o, z, o, z, z, z,
//    z, z, h, -h, -z, o, -z, -o, o, z, -z, z, m, h, -h, o, -z, -h, -o, h, -z,
//    z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, -z, h, -h, -z, -z, o, h,
//    h, f, -z, -z, -z, o, -o, o, -z, -z, -h, -o, -h, z, z, z, z, z, z, z, z, z,
//    z, z, z, -h, -z, -z, h, -o, -z, h, -h, z, f, -z, z, o, o, -z, -o, h, -z,
//    -h, o, z, z, z, z, z, z, z, z, z, z, z, z, h, z, -z, h, -h, -h, -z, o, z,
//    z, f, -z, -o, z, o, -o, o, h, -z, -h, z, z, z, z, z, z, z, z, z, z, z, z,
//    z, -h, -h, -z, -h, h, -o, -z, z, -z, z, f, z, o, o, o, h, -o, h, -z, -z,
//    -h, h, -z, z, z, z, z, z, z, z, z, z, z, z, z, -z, h, o, -h, o, o, -o, -z,
//    f, -z, -z, -z, -z, -o, -h, h, h, -z, -z, -h, z, z, z, z, z, z, z, z, z, z,
//    z, z, -h, -z, -h, -o, -o, o, -z, o, z, f, -z, z, o, -z, h, h, -h, z, -z,
//    -h, z, z, z, z, z, z, z, z, z, z, z, z, -o, h, -z, h, o, z, o, o, z, z, f,
//    -z, h, -h, -z, -o, z, h, h, -z, z, z, z, z, z, z, z, z, z, z, z, z, h, o,
//    -h, -z, z, -o, -o, o, z, -z, z, f, -h, -h, o, -z, o, -o, -z, -o, -z, -h,
//    -h, -z, z, z, z, z, o, -z, -o, o, z, z, z, z, -z, h, o, h, -z, o, h, -h,
//    m, -z, -z, -z, o, o, -o, z, h, -z, -z, h, z, z, z, z, z, o, o, o, z, z, z,
//    z, -h, -z, h, -o, -o, -z, -h, -h, z, m, -z, z, z, o, o, -o, h, z, -z, -h,
//    z, z, z, z, o, -o, o, -z, z, z, z, z, -o, -h, -z, h, -h, h, -z, o, z, z,
//    m, -z, o, -z, o, o, z, -h, h, -z, z, z, z, z, -o, -o, z, o, z, z, z, z,
//    -h, o, -h, -z, h, h, -o, -z, z, -z, z, m;
//
//  Eigen::SparseMatrix<flo::real> expected_D = dense_D.sparseView();
//
//  EXPECT_MAT_NEAR(D, expected_D);
//}

#define INTRINSIC_DIRAC(NAME)                                                  \
  TEST(IntrinsicDirac, NAME)                                                   \
  {                                                                            \
    const std::string name = #NAME;                                            \
    const std::string matrix_prefix = "../matrices/" + name;                   \
    const auto& surf =                                                         \
      TestCache::get_mesh<TestCache::HOST>("../models/" + name + ".obj");      \
    std::vector<flo::real> rho(surf.n_vertices(), 3.0f);                       \
    Eigen::Matrix<int, 1, Eigen::Dynamic> dense_valence;                       \
    Eigen::loadMarketVector(                                                   \
      dense_valence, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");  \
    gsl::span<int> valence{dense_valence.data(),                               \
                           (size_t)dense_valence.size()};                      \
    Eigen::Matrix<flo::real, 1, Eigen::Dynamic> dense_area;                    \
    Eigen::loadMarketVector(dense_area, matrix_prefix + "/area/area.mtx");     \
    gsl::span<flo::real> face_area{dense_area.data(),                          \
                                   (size_t)dense_area.size()};                 \
    auto D = flo::host::intrinsic_dirac(                                       \
      surf.vertices, surf.faces, valence, face_area, rho);                     \
    Eigen::SparseMatrix<flo::real> expected_D;                                 \
    Eigen::loadMarket(expected_D,                                              \
                      matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx"); \
    EXPECT_MAT_NEAR(D, expected_D);                                            \
  }

INTRINSIC_DIRAC(cube)
INTRINSIC_DIRAC(spot)

#undef INTRINSIC_DIRAC

