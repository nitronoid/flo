#include "test_common.h"

#include "flo/host/cotangent_laplacian.hpp"

TEST(CotangentLaplacian, cube)
{
  auto cube = make_cube();

  auto L = flo::host::cotangent_laplacian(cube.vertices, cube.faces);

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
  Eigen::SparseMatrix<flo::real> expected_L = dense_L.sparseView();

  EXPECT_MAT_NEAR(L, expected_L);

}



