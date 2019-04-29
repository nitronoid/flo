#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include <cusp/transpose.h>
#include "flo/device/vertex_mass.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read all our dependencies from disk
  auto d_area = read_device_vector<flo::real>(mp + "/face_area/face_area.mtx");
  auto d_triangle_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_triangle_adjacency/cumulative_valence.mtx");
  auto d_triangle_adjacency =
    read_device_vector<int>(mp + "/vertex_triangle_adjacency/adjacency.mtx");


  // Declare device side arrays to dump our results
  DeviceVectorR d_vertex_mass(surf.n_vertices());

  for (auto _ : state)
  {
    flo::device::vertex_mass(d_area,
                             d_triangle_adjacency,
                             {d_triangle_cumulative_valence.begin() + 1,
                              d_triangle_cumulative_valence.end()},
                             d_vertex_mass);
  }
}
}  // namespace

#define FLO_VERTEX_MASS_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_vertex_mass_##NAME(benchmark::State& state) \
  {                                                              \
    bench_impl(#NAME, state);                                    \
  }                                                              \
  BENCHMARK(DEVICE_vertex_mass_##NAME);

FLO_VERTEX_MASS_DEVICE_BENCHMARK(cube)
FLO_VERTEX_MASS_DEVICE_BENCHMARK(spot)
FLO_VERTEX_MASS_DEVICE_BENCHMARK(bunny)

#undef FLO_VERTEX_MASS_DEVICE_BENCHMARK



