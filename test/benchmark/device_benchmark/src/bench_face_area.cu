#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/face_area.cuh"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Declare device side arrays to dump our results
  DeviceVectorR d_area(surf.n_faces());

  for (auto _ : state)
  {
    flo::device::face_area(surf.vertices, surf.faces, d_area);
  }
}
}  // namespace

#define FLO_FACE_AREA_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_face_area_##NAME(benchmark::State& state) \
  {                                                            \
    bench_impl(#NAME, state);                                  \
  }                                                            \
  BENCHMARK(DEVICE_face_area_##NAME);

FLO_FACE_AREA_DEVICE_BENCHMARK(cube)
FLO_FACE_AREA_DEVICE_BENCHMARK(spot)
FLO_FACE_AREA_DEVICE_BENCHMARK(bunny)
FLO_FACE_AREA_DEVICE_BENCHMARK(tyra)

#undef FLO_FACE_AREA_DEVICE_BENCHMARK



