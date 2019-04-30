#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/host/intrinsic_dirac.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  auto rho = read_vector<flo::real>(mp + "/project_basis/rho.mtx");
  auto vertex_valence =
    read_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto face_area = read_vector<flo::real>(mp + "/face_area/face_area.mtx");
  Eigen::SparseMatrix<flo::real> D;

  for (auto _ : state)
  {
    flo::host::intrinsic_dirac(
      surf.vertices, surf.faces, vertex_valence, face_area, rho, D);
  }
}
}  // namespace

#define FLO_INTRINSIC_DIRAC_HOST_BENCHMARK(NAME)                   \
  static void HOST_intrinsic_dirac_##NAME(benchmark::State& state) \
  {                                                                \
    bench_impl(#NAME, state);                                      \
  }                                                                \
  BENCHMARK(HOST_intrinsic_dirac_##NAME);

FLO_INTRINSIC_DIRAC_HOST_BENCHMARK(cube)
FLO_INTRINSIC_DIRAC_HOST_BENCHMARK(spot)
FLO_INTRINSIC_DIRAC_HOST_BENCHMARK(bunny)
FLO_INTRINSIC_DIRAC_HOST_BENCHMARK(tyra)

#undef FLO_INTRINSIC_DIRAC_HOST_BENCHMARK
