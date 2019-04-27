#include <benchmark/benchmark.h>
#include "test_common.h"
#include <igl/cotmatrix.h>

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  Eigen::SparseMatrix<flo::real> L;
  for (auto _ : state)
  {
    igl::cotmatrix(surf.vertices, surf.faces, L);
    L = -(L.eval());
  }
}
}  // namespace

#define FLO_COTANGENT_LAPLACIAN_HOST_BENCHMARK(NAME)                   \
  static void HOST_cotangent_laplacian_##NAME(benchmark::State& state) \
  {                                                                    \
    bench_impl(#NAME, state);                                          \
  }                                                                    \
  BENCHMARK(HOST_cotangent_laplacian_##NAME);

FLO_COTANGENT_LAPLACIAN_HOST_BENCHMARK(cube)
FLO_COTANGENT_LAPLACIAN_HOST_BENCHMARK(spot)
FLO_COTANGENT_LAPLACIAN_HOST_BENCHMARK(bunny)

#undef FLO_COTANGENT_LAPLACIAN_HOST_BENCHMARK
