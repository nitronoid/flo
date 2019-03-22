#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/area.cuh"

namespace
{
void bench_impl(std::string name, benchmark::State& state)
{
  // Load our surface from the cache
  auto surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  cusp::array1d<flo::real, cusp::device_memory> d_area(surf.n_faces());
  for (auto _ : state)
  {
    flo::device::area(surf.vertices, surf.faces, d_area);
  }
}
}  // namespace

#define FLO_FACE_AREA_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_face_area_##NAME(benchmark::State& state) \
  {                                                            \
    bench_impl(#NAME, state);                                   \
  }                                                            \
  BENCHMARK(DEVICE_face_area_##NAME);

FLO_FACE_AREA_DEVICE_BENCHMARK(cube)
FLO_FACE_AREA_DEVICE_BENCHMARK(spot)
FLO_FACE_AREA_DEVICE_BENCHMARK(dense_sphere_400x400)
FLO_FACE_AREA_DEVICE_BENCHMARK(dense_sphere_1000x1000)

#undef FLO_FACE_AREA_DEVICE_BENCHMARK

