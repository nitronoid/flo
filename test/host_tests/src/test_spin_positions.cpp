#include "test_common.h"

#include "flo/host/spin_positions.hpp"
#include "flo/host/flo_matrix_operation.hpp"

TEST(SpinPositions, cube)
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
  auto ql = flo::host::to_real_quaternion_matrix(L);

  // Make this positive semi-definite by removing last row and col
  ql.conservativeResize(ql.rows() - 4, ql.cols() - 4);

  // Make this positive semi-definite by removing last edge
  using quat_t = Eigen::Matrix<flo::real, 4, 1>;
  std::vector<quat_t> edges(7);
  edges[0] = quat_t{-0.021560, -0.021560, -0.184704,  0.000000};
  edges[1] = quat_t{-0.032538,  0.032539,  0.044884, -0.000000};
  edges[2] = quat_t{ 0.032538, -0.032539,  0.044884, -0.000000};
  edges[3] = quat_t{ 0.021560,  0.021560, -0.184704,  0.000000};
  edges[4] = quat_t{-0.021560,  0.021560,  0.184704,  0.000000};
  edges[5] = quat_t{-0.032538, -0.032539, -0.044884, -0.000000};
  edges[6] = quat_t{ 0.032538,  0.032539, -0.044884, -0.000000};

  auto positions = flo::host::spin_positions(ql, edges);

  std::vector<Eigen::Matrix<flo::real, 3, 1>> expected_positions {
    {-0.062869, -0.062867, -0.996040},
    {-0.148760,  0.148766, -0.097901},
    { 0.148760, -0.148766, -0.097901},
    { 0.062869,  0.062867, -0.996040},
    {-0.062869,  0.062867,  0.996040},
    {-0.148760, -0.148766,  0.097901},
    { 0.148760,  0.148766,  0.097901},
    { 0.062869, -0.062867,  0.996040}};

  using namespace testing;
  EXPECT_THAT(positions, Pointwise(EigenNear(), expected_positions));
}





