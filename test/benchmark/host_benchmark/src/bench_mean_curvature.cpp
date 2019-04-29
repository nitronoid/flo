#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/mean_curvature.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto M = read_vector<flo::real>(mp + "/vertex_mass/vertex_mass.mtx");
  auto N =
    read_dense_matrix<flo::real, 3>(mp + "/vertex_normals/vertex_normals.mtx");
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> SH;
  for (auto _ : state)
  {
    flo::host::signed_mean_curvature(surf.vertices, L, M, N, SH);
  }
}
}  // namespace

#define FLO_MEAN_CURVATURE_HOST_BENCHMARK(NAME)                   \
  static void HOST_mean_curvature_##NAME(benchmark::State& state) \
  {                                                               \
    bench_impl(#NAME, state);                                     \
  }                                                               \
  BENCHMARK(HOST_mean_curvature_##NAME);

FLO_MEAN_CURVATURE_HOST_BENCHMARK(cube)
FLO_MEAN_CURVATURE_HOST_BENCHMARK(spot)
FLO_MEAN_CURVATURE_HOST_BENCHMARK(bunny)

#undef FLO_MEAN_CURVATURE_HOST_BENCHMARK

