#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/vertex_triangle_adjacency.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Declare device side arrays to dump our results
  DeviceVectorI d_adjacency(surf.n_faces() * 3);
  DeviceVectorI d_adjacency_keys(surf.n_faces() * 3);
  DeviceVectorI d_valence(surf.n_vertices());
  DeviceVectorI d_cumulative_valence(surf.n_vertices() + 1);

  for (auto _ : state)
  {
    flo::device::vertex_triangle_adjacency(
      surf.faces,
      d_adjacency_keys,
      d_adjacency,
      d_valence,
      d_cumulative_valence.subarray(1, surf.n_vertices()));
  }
}
}  // namespace

#define FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_vertex_triangle_adjacency_##NAME(benchmark::State& state) \
  {                                                                            \
    bench_impl(#NAME, state);                                                  \
  }                                                                            \
  BENCHMARK(DEVICE_vertex_triangle_adjacency_##NAME);

FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK(cube)
FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK(spot)
FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK(bunny)
FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK(tyra)

#undef FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK



