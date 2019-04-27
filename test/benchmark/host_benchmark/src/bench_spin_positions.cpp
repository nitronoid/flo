#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/spin_positions.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  auto L = read_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto QL = flo::host::to_real_quaternion_matrix(L);
  // Make this positive semi-definite by removing last row and col
  QL.conservativeResize(QL.rows() - 4, QL.cols() - 4);
  auto E = read_dense_matrix<flo::real, 4>(mp + "/divergent_edges/edges.mtx");
  // Make this positive semi-definite by removing last edge
  E.conservativeResize(E.rows() - 1, Eigen::NoChange);

  Eigen::Matrix<flo::real, Eigen::Dynamic, 4> V;
  for (auto _ : state)
  {
    flo::host::spin_positions(QL, E, V);
  }
}
}  // namespace

#define FLO_SPIN_POSITIONS_HOST_BENCHMARK(NAME)                   \
  static void HOST_spin_positions_##NAME(benchmark::State& state) \
  {                                                               \
    bench_impl(#NAME, state);                                     \
  }                                                               \
  BENCHMARK(HOST_spin_positions_##NAME);

FLO_SPIN_POSITIONS_HOST_BENCHMARK(cube)
FLO_SPIN_POSITIONS_HOST_BENCHMARK(spot)
FLO_SPIN_POSITIONS_HOST_BENCHMARK(bunny)

#undef FLO_SPIN_POSITIONS_HOST_BENCHMARK

