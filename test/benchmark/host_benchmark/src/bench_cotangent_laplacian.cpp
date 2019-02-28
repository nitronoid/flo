#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/valence.hpp"

#define HOST_BM_CL(BM_NAME, FILE_NAME)                              \
  static void BM_NAME(benchmark::State& state)                      \
  {                                                                 \
    auto surf = TestCache::get_mesh<TestCache::HOST>(FILE_NAME);    \
    for (auto _ : state)                                            \
    {                                                               \
      benchmark::DoNotOptimize(                                     \
        flo::host::cotangent_laplacian(surf.vertices, surf.faces)); \
    }                                                               \
  }                                                                 \
  BENCHMARK(BM_NAME)

HOST_BM_CL(HOST_cotangent_laplacian_cube, "cube.obj");
HOST_BM_CL(HOST_cotangent_laplacian_spot, "spot.obj");
HOST_BM_CL(HOST_cotangent_laplacian_sphere_400,
           "dense_sphere_400x400.obj");
HOST_BM_CL(HOST_cotangent_laplacian_sphere_1000,
           "dense_sphere_1000x1000.obj");
// HOST_BM_CL(HOST_cotangent_laplacian_sphere_1500,
// "dense_sphere_1500x1500.obj");
// HOST_BM_CL(HOST_cotangent_laplacian_cube_1000, "cube_1k.obj");

