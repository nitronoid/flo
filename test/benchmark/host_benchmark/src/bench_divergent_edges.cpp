#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/divergent_edges.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto X = read_dense_matrix<flo::real, 4>(mp + "/similarity_xform/lambda.mtx");
  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> E;
  for (auto _ : state)
  {
    flo::host::divergent_edges(surf.vertices, surf.faces, X, L, E);
  }
}
}  // namespace

#define FLO_DIVERGENT_EDGES_HOST_BENCHMARK(NAME)                   \
  static void HOST_divergent_edges_##NAME(benchmark::State& state) \
  {                                                                    \
    bench_impl(#NAME, state);                                          \
  }                                                                    \
  BENCHMARK(HOST_divergent_edges_##NAME);

FLO_DIVERGENT_EDGES_HOST_BENCHMARK(cube)
FLO_DIVERGENT_EDGES_HOST_BENCHMARK(spot)
FLO_DIVERGENT_EDGES_HOST_BENCHMARK(bunny)
FLO_DIVERGENT_EDGES_HOST_BENCHMARK(tyra)

#undef FLO_DIVERGENT_EDGES_HOST_BENCHMARK

