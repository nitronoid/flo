#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/valence.hpp"
#include "flo/host/area.hpp"

#define HOST_BM_ID(BM_NAME, FILE_NAME)                           \
  static void BM_NAME(benchmark::State& state)                   \
  {                                                              \
    auto surf = TestCache::get_mesh<TestCache::HOST>(FILE_NAME); \
    std::vector<flo::real> rho(surf.n_vertices(), 3.0f);         \
    auto face_area = flo::host::area(surf.vertices, surf.faces); \
    auto valence = flo::host::valence(surf.faces);               \
    for (auto _ : state)                                         \
    {                                                            \
      benchmark::DoNotOptimize(flo::host::intrinsic_dirac(       \
        surf.vertices, surf.faces, valence, face_area, rho));    \
    }                                                            \
  }                                                              \
  BENCHMARK(BM_NAME)

HOST_BM_ID(HOST_intrinsic_dirac_cube, "cube.obj");
HOST_BM_ID(HOST_intrinsic_dirac_spot, "spot.obj");
HOST_BM_ID(HOST_intrinsic_dirac_sphere_400, "dense_sphere_400x400.obj");
//HOST_BM_ID(HOST_intrinsic_dirac_sphere_1000, "dense_sphere_1000x1000.obj");
// HOST_BM_CL(HOST_cotangent_laplacian_sphere_1500,
// "dense_sphere_1500x1500.obj");
// HOST_BM_CL(HOST_cotangent_laplacian_cube_1000, "cube_1k.obj");

