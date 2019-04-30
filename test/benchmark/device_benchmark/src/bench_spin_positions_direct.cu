#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include <cusp/transpose.h>
#include "flo/device/spin_positions_direct.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read all our dependencies from disk
  auto d_ET =
    read_device_dense_matrix<flo::real>(mp + "/divergent_edges/edges.mtx");
  auto d_LQ = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/quaternion_cotangent_laplacian.mtx");
  cusp::array2d<flo::real, cusp::device_memory> d_E(d_ET.num_cols,
                                                    d_ET.num_rows);
  cusp::transpose(d_ET, d_E);


  // Declare device side arrays to dump our results
  DeviceDenseMatrixR d_vertices(4, surf.n_vertices(), 0.f);

  for (auto _ : state)
  {
    flo::device::direct::spin_positions(d_LQ, d_E, d_vertices);
  }
}
}  // namespace

#define FLO_SPIN_POSITIONS_DIRECT_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_spin_positions_direct_##NAME(benchmark::State& state) \
  {                                                                        \
    bench_impl(#NAME, state);                                              \
  }                                                                        \
  BENCHMARK(DEVICE_spin_positions_direct_##NAME);

FLO_SPIN_POSITIONS_DIRECT_DEVICE_BENCHMARK(spot)
FLO_SPIN_POSITIONS_DIRECT_DEVICE_BENCHMARK(bunny)
FLO_SPIN_POSITIONS_DIRECT_DEVICE_BENCHMARK(tyra)

#undef FLO_SPIN_POSITIONS_DIRECT_DEVICE_BENCHMARK


