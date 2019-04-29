#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/quaternion_matrix.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read all our dependencies from disk
  auto d_L = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  // Declare device side arrays to dump our results
  DeviceSparseMatrixR d_QL(
    surf.n_vertices() * 4, surf.n_vertices() * 4, d_L.values.size() * 16);

  for (auto _ : state)
  {
    flo::device::to_real_quaternion_matrix(
      d_L, {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()}, d_QL);
  }
}
}  // namespace

#define FLO_QUATERNION_MATRIX_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_quaternion_matrix_##NAME(benchmark::State& state) \
  {                                                                    \
    bench_impl(#NAME, state);                                          \
  }                                                                    \
  BENCHMARK(DEVICE_quaternion_matrix_##NAME);

FLO_QUATERNION_MATRIX_DEVICE_BENCHMARK(cube)
FLO_QUATERNION_MATRIX_DEVICE_BENCHMARK(spot)
FLO_QUATERNION_MATRIX_DEVICE_BENCHMARK(bunny)

#undef FLO_QUATERNION_MATRIX_DEVICE_BENCHMARK




