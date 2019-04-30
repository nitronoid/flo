#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/similarity_xform.cuh"
#include "flo/device/cu_raii.cuh"

namespace
{
void bench_impl(std::string name, benchmark::State& state)
{
  const std::string matrix_prefix = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read intrinsic dirac matrix
  auto d_D = read_device_sparse_matrix<flo::real>(
    matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx");

  // Copy to device
  flo::device::cu_raii::solver::SolverSp solver;
  flo::device::cu_raii::sparse::Handle sparse_handle;

  DeviceDenseMatrixR d_xform(4, surf.n_vertices());
  for (auto _ : state)
  {
    flo::device::direct::similarity_xform(
        &sparse_handle, &solver, d_D, d_xform);
  }
}
}  // namespace

#define FLO_SIMILARITY_XFORM_DIRECT_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_similarity_xform_direct_##NAME(benchmark::State& state) \
  {                                                                          \
    bench_impl(#NAME, state);                                                \
  }                                                                          \
  BENCHMARK(DEVICE_similarity_xform_direct_##NAME);

FLO_SIMILARITY_XFORM_DIRECT_DEVICE_BENCHMARK(cube)
FLO_SIMILARITY_XFORM_DIRECT_DEVICE_BENCHMARK(spot)
FLO_SIMILARITY_XFORM_DIRECT_DEVICE_BENCHMARK(bunny)
FLO_SIMILARITY_XFORM_DIRECT_DEVICE_BENCHMARK(tyra)

#undef FLO_SIMILARITY_XFORM_DIRECT_DEVICE_BENCHMARK
