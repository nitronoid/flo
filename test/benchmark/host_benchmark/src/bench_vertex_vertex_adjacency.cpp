#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/vertex_vertex_adjacency.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  // Declare arrays to dump our results
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVAK;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVA;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVV;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VVCV;
  for (auto _ : state)
  {
    flo::host::vertex_vertex_adjacency(surf.faces, VVAK, VVA, VVV, VVCV);
  }
}
}  // namespace

#define FLO_VERTEX_VERTEX_ADJACENCY_HOST_BENCHMARK(NAME)                   \
  static void HOST_vertex_vertex_adjacency_##NAME(benchmark::State& state) \
  {                                                                        \
    bench_impl(#NAME, state);                                              \
  }                                                                        \
  BENCHMARK(HOST_vertex_vertex_adjacency_##NAME);

FLO_VERTEX_VERTEX_ADJACENCY_HOST_BENCHMARK(cube)
FLO_VERTEX_VERTEX_ADJACENCY_HOST_BENCHMARK(spot)
FLO_VERTEX_VERTEX_ADJACENCY_HOST_BENCHMARK(bunny)

#undef FLO_VERTEX_VERTEX_ADJACENCY_HOST_BENCHMARK

