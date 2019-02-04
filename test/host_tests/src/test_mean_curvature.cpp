#include "test_common.h"

#include "flo/host/mean_curvature.hpp"

TEST(MeanCurvature, cube)
{
  auto cube = make_cube();

  Eigen::Matrix<flo::real, 8, 8> dense_L(8,8);
  dense_L <<
     3, -1, -1,  0, -0,  0, -1, -0, 
    -1,  3, -0, -1,  0,  0,  0, -1, 
    -1, -0,  3, -1, -1,  0,  0,  0, 
     0, -1, -1,  3, -0, -1,  0, -0, 
    -0,  0, -1, -0,  3, -1, -1,  0, 
     0,  0,  0, -1, -1,  3, -0, -1, 
    -1,  0,  0,  0, -1, -0,  3, -1, 
    -0, -1,  0, -0,  0, -1, -1,  3;
  Eigen::SparseMatrix<flo::real> L = dense_L.sparseView();

  using normal_t = Eigen::Matrix<flo::real, 3, 1>;
  std::vector<normal_t> normals {
    {-0.57735, -0.57735,  0.57735},
    { 0.57735, -0.57735,  0.57735},
    {-0.57735,  0.57735,  0.57735},
    { 0.57735,  0.57735,  0.57735},
    {-0.57735,  0.57735, -0.57735},
    { 0.57735,  0.57735, -0.57735},
    {-0.57735, -0.57735, -0.57735},
    { 0.57735, -0.57735, -0.57735}};

  std::vector<flo::real> mass {
    0.833333, 0.666667, 0.666667, 0.833333, 0.833333, 0.666667, 0.666667, 0.833333};


  auto HN = flo::host::mean_curvature_normal(cube.vertices, L, mass);

  std::vector<normal_t> expected_HN {
    { 0.20,  0.20, -0.20},
    {-0.25,  0.25, -0.25},
    { 0.25, -0.25, -0.25},
    {-0.20, -0.20, -0.20},
    { 0.20, -0.20,  0.20},
    {-0.25, -0.25,  0.25},
    { 0.25,  0.25,  0.25},
    {-0.20,  0.20,  0.20}};

  auto H  = flo::host::mean_curvature(cube.vertices, L, mass);
  std::vector<flo::real> expected_H {
    0.34641, 0.43301, 0.43301, 0.34641, 0.34641, 0.43301, 0.43301, 0.34641};

  // Should be all positive like H
  auto SH = flo::host::signed_mean_curvature(cube.vertices, L, mass, normals);

  using namespace testing;
  EXPECT_THAT(expected_HN, Pointwise(EigenNear(), HN));
  EXPECT_THAT(expected_H,  Pointwise(FloatNear(FLOAT_SOFT_EPSILON), H));
  EXPECT_THAT(expected_H,  Pointwise(FloatNear(FLOAT_SOFT_EPSILON), SH));
}

