#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/host/similarity_xform.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  auto D =
    read_sparse_matrix<flo::real>(mp + "/intrinsic_dirac/intrinsic_dirac.mtx");
  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> X;
  for (auto _ : state)
  {
    flo::host::similarity_xform(D, X);
  }
}
}  // namespace

#define FLO_SIMILARITY_XFORM_HOST_BENCHMARK(NAME)                   \
  static void HOST_similarity_xform_##NAME(benchmark::State& state) \
  {                                                                 \
    bench_impl(#NAME, state);                                       \
  }                                                                 \
  BENCHMARK(HOST_similarity_xform_##NAME);

FLO_SIMILARITY_XFORM_HOST_BENCHMARK(cube)
FLO_SIMILARITY_XFORM_HOST_BENCHMARK(spot)
FLO_SIMILARITY_XFORM_HOST_BENCHMARK(bunny)

#undef FLO_SIMILARITY_XFORM_HOST_BENCHMARK
