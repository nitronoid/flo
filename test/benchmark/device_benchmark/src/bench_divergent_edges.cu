#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/divergent_edges.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read all our dependencies from disk
  auto d_xform =
    read_device_dense_matrix<flo::real>(mp + "/similarity_xform/lambda.mtx");
  auto d_L = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");

  // Declare device side arrays to dump our results
  DeviceDenseMatrixR d_edges(4, d_xform.num_rows);

  for (auto _ : state)
  {
    flo::device::divergent_edges(
      surf.vertices, surf.faces, d_xform.values, d_L, d_edges);
  }
}
}  // namespace

#define FLO_DIVERGENT_EDGES_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_divergent_edges_##NAME(benchmark::State& state) \
  {                                                                  \
    bench_impl(#NAME, state);                                        \
  }                                                                  \
  BENCHMARK(DEVICE_divergent_edges_##NAME);

FLO_DIVERGENT_EDGES_DEVICE_BENCHMARK(cube)
FLO_DIVERGENT_EDGES_DEVICE_BENCHMARK(spot)
FLO_DIVERGENT_EDGES_DEVICE_BENCHMARK(bunny)

#undef FLO_DIVERGENT_EDGES_DEVICE_BENCHMARK


