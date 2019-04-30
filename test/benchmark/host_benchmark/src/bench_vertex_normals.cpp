#include <benchmark/benchmark.h>
#include "test_common.h"
#include <igl/per_vertex_normals.h>

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  Eigen::Matrix<flo::real, Eigen::Dynamic, 3> N;
  for (auto _ : state)
  {
    using namespace igl;
    per_vertex_normals(
      surf.vertices, surf.faces, PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);
  }
}
}  // namespace

#define FLO_VERTEX_NORMALS_HOST_BENCHMARK(NAME)                   \
  static void HOST_vertex_normals_##NAME(benchmark::State& state) \
  {                                                                    \
    bench_impl(#NAME, state);                                          \
  }                                                                    \
  BENCHMARK(HOST_vertex_normals_##NAME);

FLO_VERTEX_NORMALS_HOST_BENCHMARK(cube)
FLO_VERTEX_NORMALS_HOST_BENCHMARK(spot)
FLO_VERTEX_NORMALS_HOST_BENCHMARK(bunny)
FLO_VERTEX_NORMALS_HOST_BENCHMARK(tyra)

#undef FLO_VERTEX_NORMALS_HOST_BENCHMARK

