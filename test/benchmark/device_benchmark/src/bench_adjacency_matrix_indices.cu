#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/adjacency_matrix_indices.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read all our dependencies from disk
  auto d_adjacency_keys =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto d_adjacency =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto d_valence =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");

  // Declare device side arrays to dump our results
  DeviceDenseMatrixI d_indices(6, surf.n_faces());

  for (auto _ : state)
  {
    flo::device::adjacency_matrix_indices(
      surf.faces, d_adjacency, d_cumulative_valence, d_indices);
  }
}
}  // namespace

#define FLO_ADJACENCY_MATRIX_INDICES_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_adjacency_matrix_indices_##NAME(benchmark::State& state) \
  {                                                                           \
    bench_impl(#NAME, state);                                                 \
  }                                                                           \
  BENCHMARK(DEVICE_adjacency_matrix_indices_##NAME);

FLO_ADJACENCY_MATRIX_INDICES_DEVICE_BENCHMARK(cube)
FLO_ADJACENCY_MATRIX_INDICES_DEVICE_BENCHMARK(spot)
FLO_ADJACENCY_MATRIX_INDICES_DEVICE_BENCHMARK(bunny)

#undef FLO_ADJACENCY_MATRIX_INDICES_DEVICE_BENCHMARK

