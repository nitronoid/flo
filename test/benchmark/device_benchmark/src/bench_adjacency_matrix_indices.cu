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
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");

  // Declare device side arrays to dump our results
  DeviceDenseMatrixI d_entry_indices(6, surf.n_faces());
  DeviceVectorI d_diagonal_indices(surf.n_vertices());
  DeviceSparseMatrixR d_L(surf.n_vertices(),
                          surf.n_vertices(),
                          d_cumulative_valence.back() + surf.n_vertices());

  auto temp_ptr = thrust::device_pointer_cast(
      reinterpret_cast<void*>(d_L.values.begin().base().get()));

  for (auto _ : state)
  {
    flo::device::adjacency_matrix_indices(surf.faces,
                                          d_adjacency_keys,
                                          d_adjacency,
                                          d_cumulative_valence, 
                                          d_entry_indices,
                                          d_diagonal_indices,
                                          d_L.row_indices,
                                          d_L.column_indices,
                                          temp_ptr);
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
FLO_ADJACENCY_MATRIX_INDICES_DEVICE_BENCHMARK(tyra)

#undef FLO_ADJACENCY_MATRIX_INDICES_DEVICE_BENCHMARK

