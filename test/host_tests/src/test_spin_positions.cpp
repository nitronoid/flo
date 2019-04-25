#include "test_common.h"
#include "flo/host/spin_positions.hpp"
#include "flo/host/flo_matrix_operation.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto QL = flo::host::to_real_quaternion_matrix(L);
  // Make this positive semi-definite by removing last row and col
  QL.conservativeResize(QL.rows() - 4, QL.cols() - 4);

  // Make this positive semi-definite by removing last edge
  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> E(7, 4);
  // clang-format off
  E <<
    -0.021560, -0.021560, -0.184704,  0.000000,
    -0.032538,  0.032539,  0.044884, -0.000000,
     0.032538, -0.032539,  0.044884, -0.000000,
     0.021560,  0.021560, -0.184704,  0.000000,
    -0.021560,  0.021560,  0.184704,  0.000000,
    -0.032538, -0.032539, -0.044884, -0.000000,
     0.032538,  0.032539, -0.044884, -0.000000;
  // clang-format on

  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> V;
  flo::host::spin_positions(QL, E, V);
  std::cout<<"pass\n";

  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> expected_V(8, 4);
  // clang-format off
  expected_V <<
    -0.062869, -0.062867, -0.996040,  0.000000,
    -0.148760,  0.148766, -0.097901,  0.000000,
     0.148760, -0.148766, -0.097901,  0.000000,
     0.062869,  0.062867, -0.996040,  0.000000,
    -0.062869,  0.062867,  0.996040,  0.000000,
    -0.148760, -0.148766,  0.097901,  0.000000,
     0.148760,  0.148766,  0.097901,  0.000000,
     0.062869, -0.062867,  0.996040,  0.000000;
  // clang-format on

  EXPECT_MAT_NEAR(V, expected_V);
}
}

#define FLO_SPIN_POSITIONS_TEST(NAME) \
  TEST(SpinPositions, NAME)           \
  {                                   \
    test(#NAME);                      \
  }

//FLO_SPIN_POSITIONS_TEST(cube)
// FLO_SPIN_POSITIONS_TEST(spot)

#undef FLO_SPIN_POSITIONS_TEST





