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
  auto d_valence =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_adjacency_keys =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto d_adjacency =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto d_offsets =
    read_device_dense_matrix<int>(mp + "/adjacency_matrix_offset/offsets.mtx");

  // Allocate a sparse matrix to store our result
  DeviceSparseMatrixR d_L(surf.n_vertices(),
                          surf.n_vertices(),
                          d_cumulative_valence.back() + surf.n_vertices());

  // Allocate a dense 1 dimensional array to receive diagonal element indices
  DeviceVectorI d_diagonals(surf.n_vertices());
  for (auto _ : state)
  {
    flo::device::cotangent_laplacian(surf.vertices,
                                     surf.faces,
                                     d_offsets,
                                     d_adjacency_keys,
                                     d_adjacency,
                                     d_cumulative_valence,
                                     d_diagonals,
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

#undef FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK
