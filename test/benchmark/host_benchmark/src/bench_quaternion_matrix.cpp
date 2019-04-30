#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/flo_matrix_operation.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");

  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> E;
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(flo::host::to_real_quaternion_matrix(L));
  }
}
}  // namespace

#define FLO_QUATERNION_MATRIX_HOST_BENCHMARK(NAME)                   \
  static void HOST_quaternion_matrix_##NAME(benchmark::State& state) \
  {                                                                  \
    bench_impl(#NAME, state);                                        \
  }                                                                  \
  BENCHMARK(HOST_quaternion_matrix_##NAME);

FLO_QUATERNION_MATRIX_HOST_BENCHMARK(cube)
FLO_QUATERNION_MATRIX_HOST_BENCHMARK(spot)
FLO_QUATERNION_MATRIX_HOST_BENCHMARK(bunny)
FLO_QUATERNION_MATRIX_HOST_BENCHMARK(tyra)

#undef FLO_QUATERNION_MATRIX_HOST_BENCHMARK


