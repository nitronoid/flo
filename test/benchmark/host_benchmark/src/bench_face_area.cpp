#include <benchmark/benchmark.h>
#include "test_common.h"
#include <igl/doublearea.h>

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  const std::string mp = "../../matrices/" + name;
  auto surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");
  Eigen::Matrix<flo::real, Eigen::Dynamic, 1> A;
  for (auto _ : state)
  {
    igl::doublearea(surf.vertices, surf.faces, A);
    A *= 0.5f;
  }
}
}  // namespace

#define FLO_FACE_AREA_HOST_BENCHMARK(NAME)                   \
  static void HOST_face_area_##NAME(benchmark::State& state) \
  {                                                          \
    bench_impl(#NAME, state);                                \
  }                                                          \
  BENCHMARK(HOST_face_area_##NAME);

FLO_FACE_AREA_HOST_BENCHMARK(cube)
FLO_FACE_AREA_HOST_BENCHMARK(spot)
FLO_FACE_AREA_HOST_BENCHMARK(bunny)

#undef FLO_FACE_AREA_HOST_BENCHMARK
