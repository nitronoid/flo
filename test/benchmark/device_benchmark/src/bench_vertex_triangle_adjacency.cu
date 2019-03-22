#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/vertex_triangle_adjacency.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Load our surface from the cache
  auto surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  cusp::array1d<int, cusp::device_memory> d_adjacency(surf.n_faces() * 3);
  cusp::array1d<int, cusp::device_memory> d_valence(surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
    surf.n_vertices() + 1);
  thrust::device_vector<int> temp(surf.n_faces() * 3);
  for (auto _ : state)
  {
    flo::device::vertex_triangle_adjacency(
      surf.faces,
      temp.data(),
      d_adjacency,
      d_valence,
      {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()});
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
FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK(dense_sphere_400x400)
FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK(dense_sphere_1000x1000)

#undef FLO_VERTEX_TRIANGLE_ADJACENCY_DEVICE_BENCHMARK

