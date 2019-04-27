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
  auto h_D = read_host_sparse_matrix<flo::real>(
    matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx");

  // Copy to device
  DeviceSparseMatrixR d_Dr = h_D;
  flo::device::cu_raii::solver::SolverSp solver;
  flo::device::cu_raii::sparse::Handle sparse_handle;

  DeviceDenseMatrixR d_xform(4, surf.n_vertices());
  for (auto _ : state)
  {
    flo::device::similarity_xform(&sparse_handle, &solver, d_Dr, d_xform);
  }
}
}  // namespace

#define FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_similarity_xform_##NAME(benchmark::State& state) \
  {                                                                   \
    bench_impl(#NAME, state);                                         \
  }                                                                   \
  BENCHMARK(DEVICE_similarity_xform_##NAME);

FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(cube)
FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(spot)
FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(bunny)

#undef FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK
