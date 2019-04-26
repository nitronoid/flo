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
  auto N =
    read_dense_matrix<flo::real, 3>(mp + "/vertex_normals/vertex_normals.mtx");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> HN;
  flo::host::mean_curvature_normal(surf.vertices, L, M, HN);

  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> H;
  flo::host::mean_curvature(surf.vertices, L, M, H);

  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> SH;
  flo::host::signed_mean_curvature(surf.vertices, L, M, N, SH);

  auto expected_HN = read_dense_matrix<flo::real, 3>(
    mp + "/mean_curvature/mean_curvature_normal.mtx");
  auto expected_H =
    read_vector<flo::real>(mp + "/mean_curvature/mean_curvature.mtx");
  auto expected_SH =
    read_vector<flo::real>(mp + "/mean_curvature/signed_mean_curvature.mtx");

  EXPECT_MAT_NEAR(HN, expected_HN);
  EXPECT_MAT_NEAR(H, expected_H);
  EXPECT_MAT_NEAR(SH, expected_SH);
}
}  // namespace

#define FLO_MEAN_CURVATURE_TEST(NAME) \
  TEST(MeanCurvature, NAME)           \
  {                                   \
    test(#NAME);                      \
  }

FLO_MEAN_CURVATURE_TEST(cube)
FLO_MEAN_CURVATURE_TEST(spot)
FLO_MEAN_CURVATURE_TEST(bunny)

#undef FLO_MEAN_CURVATURE_TEST
