#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/vertex_triangle_adjacency.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  // Declare arrays to dump our results
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTAK;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTA;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTV;
  Eigen::Matrix<int, Eigen::Dynamic, 1> VTCV;
  for (auto _ : state)
  {
    flo::host::vertex_triangle_adjacency(surf.faces, VTAK, VTA, VTV, VTCV);
  }
}
}  // namespace

#define FLO_VERTEX_TRIANGLE_ADJACENCY_HOST_BENCHMARK(NAME)                   \
  static void HOST_vertex_triangle_adjacency_##NAME(benchmark::State& state) \
  {                                                                          \
    bench_impl(#NAME, state);                                                \
  }                                                                          \
  BENCHMARK(HOST_vertex_triangle_adjacency_##NAME);

FLO_VERTEX_TRIANGLE_ADJACENCY_HOST_BENCHMARK(cube)
FLO_VERTEX_TRIANGLE_ADJACENCY_HOST_BENCHMARK(spot)
FLO_VERTEX_TRIANGLE_ADJACENCY_HOST_BENCHMARK(bunny)

#undef FLO_VERTEX_TRIANGLE_ADJACENCY_HOST_BENCHMARK
