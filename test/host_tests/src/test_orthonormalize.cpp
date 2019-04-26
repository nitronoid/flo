#include "test_common.h"
#include "flo/host/orthonormalize.hpp"

namespace
{
// Define an immersed inner-product using the mass matrix
struct InnerProduct
{
  InnerProduct(const Eigen::Matrix<flo::real, Eigen::Dynamic, 1>& M) : M(M)
  {
  }

  const Eigen::Matrix<flo::real, Eigen::Dynamic, 1>& M;

  flo::real
  operator()(const Eigen::Matrix<flo::real, Eigen::Dynamic, 1>& x,
             const Eigen::Matrix<flo::real, Eigen::Dynamic, 1>& y) const
  {
    auto single_mat = (x.transpose() * M.asDiagonal() * y).eval();
    return single_mat(0, 0);
  }
};

void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto M = read_vector<flo::real>(mp + "/vertex_mass/vertex_mass.mtx");
  auto N =
    read_dense_matrix<flo::real, 3>(mp + "/vertex_normals/vertex_normals.mtx");

  // Build our constraints {1, N.x, N.y, N.z}
  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> constraints(N.rows(), 4);
  constraints.col(0) =
    Eigen::Matrix<flo::real, Eigen::Dynamic, 1>::Ones(N.rows());
  constraints.col(1) = N.col(0);
  constraints.col(2) = N.col(1);
  constraints.col(3) = N.col(2);

  // Build a constraint basis using the Gram–Schmidt process
  Eigen::Matrix<flo::real, Eigen::Dynamic, Eigen::Dynamic> U;
  flo::host::orthonormalize(constraints, InnerProduct{M}, U);

  auto expected_U =
    read_dense_matrix<flo::real>(mp + "/orthonormalize/basis.mtx");

  EXPECT_MAT_NEAR(U, expected_U);
}
}  // namespace

#define FLO_ORTHONORMALIZE_TEST(NAME) \
  TEST(Orthonormalize, NAME)          \
  {                                   \
    test(#NAME);                      \
  }

FLO_ORTHONORMALIZE_TEST(cube)
FLO_ORTHONORMALIZE_TEST(spot)
FLO_ORTHONORMALIZE_TEST(bunny)

#undef FLO_ORTHONORMALIZE_TEST

