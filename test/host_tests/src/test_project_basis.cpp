#include "test_common.h"
#include "flo/host/project_basis.hpp"

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
  auto SH =
    read_vector<flo::real>(mp + "/mean_curvature/signed_mean_curvature.mtx");
  auto U = read_dense_matrix<flo::real>(mp + "/orthonormalize/basis.mtx");

  SH *= -1.f;
  flo::host::project_basis(SH, U, InnerProduct{M});

  auto expected_SH =
    read_vector<flo::real>(mp + "/project_basis/projected_mean_curvature.mtx");

  EXPECT_MAT_NEAR(SH, expected_SH);
}
}  // namespace

#define FLO_PROJECT_BASIS_TEST(NAME) \
  TEST(ProjectBasis, NAME)           \
  {                                  \
    test(#NAME);                     \
  }

FLO_PROJECT_BASIS_TEST(cube)
FLO_PROJECT_BASIS_TEST(spot)
FLO_PROJECT_BASIS_TEST(bunny)

#undef FLO_PROJECT_BASIS_TEST

