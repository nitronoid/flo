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
  auto E =
    read_dense_matrix<flo::real, 4>(mp + "/divergent_edges/edges.mtx");
  // Make this positive semi-definite by removing last edge
  E.conservativeResize(E.rows() - 1, Eigen::NoChange);

  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> V;
  flo::host::spin_positions(QL, E, V);

  auto expected_V =
    read_dense_matrix<flo::real, 4>(mp + "/spin_positions/positions.mtx");

  EXPECT_MAT_NEAR(V, expected_V);
}
}  // namespace

#define FLO_SPIN_POSITIONS_TEST(NAME) \
  TEST(SpinPositions, NAME)           \
  {                                   \
    test(#NAME);                      \
  }

FLO_SPIN_POSITIONS_TEST(cube)
FLO_SPIN_POSITIONS_TEST(spot)
FLO_SPIN_POSITIONS_TEST(bunny)

#undef FLO_SPIN_POSITIONS_TEST

