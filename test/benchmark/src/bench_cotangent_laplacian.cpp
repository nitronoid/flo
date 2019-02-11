#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/valence.hpp"
#include "flo/load_mesh.hpp"

#define HOST_BM_CL(BM_NAME, FILE_NAME)                              \
  static void BM_NAME(benchmark::State& state)                      \
  {                                                                 \
    auto surf = flo::load_mesh(FILE_NAME);                          \
    auto vv = flo::host::valence(surf.faces); \
    auto tv = std::accumulate(vv.begin(), vv.end(), 0);\
    std::cout<<tv+surf.n_vertices()<<'\n';\
    for (auto _ : state)                                            \
    {                                                               \
      benchmark::DoNotOptimize(                                     \
        flo::host::cotangent_laplacian(surf.vertices, surf.faces)); \
    }                                                               \
    auto L = flo::host::cotangent_laplacian(surf.vertices, surf.faces);\
    std::cout<<L.nonZeros()<<'\n';\
  }                                                                 \
  BENCHMARK(BM_NAME)

HOST_BM_CL(HOST_cotangent_laplacian_cube, "../models/cube.obj");
HOST_BM_CL(HOST_cotangent_laplacian_spot, "../models/spot.obj");
HOST_BM_CL(HOST_cotangent_laplacian_sphere_400,
           "../models/dense_sphere_400x400.obj");
HOST_BM_CL(HOST_cotangent_laplacian_sphere_1000,
           "../models/dense_sphere_1000x1000.obj");
// HOST_BM_CL(HOST_cotangent_laplacian_sphere_1500,
// "../models/dense_sphere_1500x1500.obj");
// HOST_BM_CL(HOST_cotangent_laplacian_cube_1000, "../models/cube_1k.obj");

