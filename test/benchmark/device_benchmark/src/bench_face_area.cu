#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/area.cuh"

#define DEVICE_BM_FA(BM_NAME, FILE_NAME)                           \
  static void BM_NAME(benchmark::State& state)                     \
  {                                                                \
    auto surf = TestCache::get_mesh<TestCache::DEVICE>(FILE_NAME); \
    for (auto _ : state)                                           \
    {                                                              \
      benchmark::DoNotOptimize(flo::device::area(                  \
        surf.vertices.data(), surf.faces.data(), surf.n_faces())); \
    }                                                              \
  }                                                                \
  BENCHMARK(BM_NAME)

DEVICE_BM_FA(DEVICE_face_area_cube_1, "cube.obj");
DEVICE_BM_FA(DEVICE_face_area_spot, "spot.obj");
DEVICE_BM_FA(DEVICE_face_area_sphere_400, "dense_sphere_400x400.obj");
DEVICE_BM_FA(DEVICE_face_area_sphere_1000, "dense_sphere_1000x1000.obj");
