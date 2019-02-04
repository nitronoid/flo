#include "test_common.h"

#include "flo/host/divergent_edges.hpp"

TEST(DivergentEdges, cube)
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

  using quat_t = Eigen::Matrix<flo::real, 4, 1>;
  std::vector<quat_t> lambda(8);
  lambda[0] = quat_t{-0.266601,  0.266601,  0.000000,  0.051751};
  lambda[1] = quat_t{ 0.159860,  0.159860,  0.000000,  0.232507};
  lambda[2] = quat_t{-0.159860, -0.159860,  0.000000,  0.232507}; 
  lambda[3] = quat_t{ 0.266601, -0.266601,  0.000000,  0.051751};
  lambda[4] = quat_t{-0.266601, -0.266601, -0.000000,  0.051751}; 
  lambda[5] = quat_t{ 0.159860, -0.159860,  0.000000,  0.232507};
  lambda[6] = quat_t{-0.159860,  0.159860, -0.000000,  0.232507}; 
  lambda[7] = quat_t{ 0.266601,  0.266601,  0.000000,  0.051751};

  auto edges = flo::host::divergent_edges(cube.vertices, cube.faces, lambda, L);

  std::vector<quat_t> expected_edges(8);
  expected_edges[0] = quat_t{-0.021560, -0.021560, -0.184704,  0.000000};
  expected_edges[1] = quat_t{-0.032538,  0.032539,  0.044884, -0.000000};
  expected_edges[2] = quat_t{ 0.032538, -0.032539,  0.044884, -0.000000};
  expected_edges[3] = quat_t{ 0.021560,  0.021560, -0.184704,  0.000000};
  expected_edges[4] = quat_t{-0.021560,  0.021560,  0.184704,  0.000000};
  expected_edges[5] = quat_t{-0.032538, -0.032539, -0.044884, -0.000000};
  expected_edges[6] = quat_t{ 0.032538,  0.032539, -0.044884, -0.000000};
  expected_edges[7] = quat_t{ 0.021560, -0.021560,  0.184704,  0.000000};

  using namespace testing;
  EXPECT_THAT(expected_edges, Pointwise(EigenNear(), edges));
}






