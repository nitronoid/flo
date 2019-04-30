#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/face_area.cuh"
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read all our dependencies from disk
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_entry_indices =
    read_device_dense_matrix<int>(mp + "/adjacency_matrix_indices/indices.mtx");
  auto d_diagonal_indices =
    read_device_vector<int>(mp + "/cotangent_laplacian/diagonals.mtx");

  // Allocate a sparse matrix to store our result
  DeviceSparseMatrixR d_L(surf.n_vertices(),
                          surf.n_vertices(),
                          d_cumulative_valence.back() + surf.n_vertices());

  for (auto _ : state)
  {
    flo::device::cotangent_laplacian_values(surf.vertices,
                                            surf.faces,
                                            d_entry_indices,
                                            d_diagonal_indices,
                                            d_L);
  }
}
}  // namespace

#define FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_cotangent_laplacian_##NAME(benchmark::State& state) \
  {                                                                      \
    bench_impl(#NAME, state);                                            \
  }                                                                      \
  BENCHMARK(DEVICE_cotangent_laplacian_##NAME);

FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(cube)
FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(spot)
FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(bunny)
FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(tyra)

#undef FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK
