#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/vertex_mass.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> M;
  for (auto _ : state)
  {
    flo::host::vertex_mass(surf.vertices, surf.faces, M);
  }
}
}  // namespace

#define FLO_VERTEX_MASS_HOST_BENCHMARK(NAME)                   \
  static void HOST_vertex_mass_##NAME(benchmark::State& state) \
  {                                                            \
    bench_impl(#NAME, state);                                  \
  }                                                            \
  BENCHMARK(HOST_vertex_mass_##NAME);

FLO_VERTEX_MASS_HOST_BENCHMARK(cube)
FLO_VERTEX_MASS_HOST_BENCHMARK(spot)
FLO_VERTEX_MASS_HOST_BENCHMARK(bunny)

#undef FLO_VERTEX_MASS_HOST_BENCHMARK
