#include "test_common.h"

#include "flo/host/mean_curvature.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto M = read_vector<flo::real>(mp + "/vertex_mass/vertex_mass.mtx");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> N(8,3);
  // clang-format off
  N <<
    -0.57735, -0.57735,  0.57735,
     0.57735, -0.57735,  0.57735,
    -0.57735,  0.57735,  0.57735,
     0.57735,  0.57735,  0.57735,
    -0.57735,  0.57735, -0.57735,
     0.57735,  0.57735, -0.57735,
    -0.57735, -0.57735, -0.57735,
     0.57735, -0.57735, -0.57735;
  // clang-format on

  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> HN;
  flo::host::mean_curvature_normal(surf.vertices, L, M, HN);

  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> H;
  flo::host::mean_curvature(surf.vertices, L, M, H);

  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> SH;
  flo::host::signed_mean_curvature(surf.vertices, L, M, N, SH);

  //-------------------------------------------------------------------------
  // RESULTS
  //-------------------------------------------------------------------------
  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> expected_HN(8,3);
  // clang-format off
  expected_HN << 
     0.20,  0.20, -0.20,
    -0.25,  0.25, -0.25,
     0.25, -0.25, -0.25,
    -0.20, -0.20, -0.20,
     0.20, -0.20,  0.20,
    -0.25, -0.25,  0.25,
     0.25,  0.25,  0.25,
    -0.20,  0.20,  0.20;
  // clang-format on
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> expected_H(8,1);
  // clang-format off
  expected_H << 
    0.34641, 0.43301, 0.43301, 0.34641, 0.34641, 0.43301, 0.43301, 0.34641;
  // clang-format on

  EXPECT_MAT_NEAR(HN, expected_HN);
  EXPECT_MAT_NEAR(H, expected_H);
  EXPECT_MAT_NEAR(SH, expected_H);
}
}

#define FLO_MEAN_CURVATURE_TEST(NAME) \
  TEST(MeanCurvature, NAME)           \
  {                                   \
    test(#NAME);                      \
  }

FLO_MEAN_CURVATURE_TEST(cube)
// FLO_MEAN_CURVATURE_TEST(spot)

#undef FLO_MEAN_CURVATURE_TEST
