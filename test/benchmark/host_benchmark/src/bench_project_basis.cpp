#include <benchmark/benchmark.h>
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

static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  auto M = read_vector<flo::real>(mp + "/vertex_mass/vertex_mass.mtx");
  auto SH =
    read_vector<flo::real>(mp + "/mean_curvature/signed_mean_curvature.mtx");
  auto U = read_dense_matrix<flo::real>(mp + "/orthonormalize/basis.mtx");
  for (auto _ : state)
  {
    flo::host::project_basis(SH, U, InnerProduct{M});
  }
}
}  // namespace

#define FLO_PROJECT_BASIS_HOST_BENCHMARK(NAME)                   \
  static void HOST_project_basis_##NAME(benchmark::State& state) \
  {                                                              \
    bench_impl(#NAME, state);                                    \
  }                                                              \
  BENCHMARK(HOST_project_basis_##NAME);

FLO_PROJECT_BASIS_HOST_BENCHMARK(cube)
FLO_PROJECT_BASIS_HOST_BENCHMARK(spot)
FLO_PROJECT_BASIS_HOST_BENCHMARK(bunny)
FLO_PROJECT_BASIS_HOST_BENCHMARK(tyra)

#undef FLO_PROJECT_BASIS_HOST_BENCHMARK
