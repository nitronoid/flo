#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/vertex_mass.hpp"

#define HOST_BM_VM(BM_NAME, FILE_NAME) \
static void BM_NAME(benchmark::State& state) \
{ \
    auto surf = TestCache::get_mesh(FILE_NAME);                                \
  for (auto _ : state) \
  { \
    benchmark::DoNotOptimize(flo::host::vertex_mass(surf.vertices, surf.faces));\
  }\
}\
BENCHMARK(BM_NAME)

HOST_BM_VM(HOST_vertex_mass_cube, "../models/cube.obj");
HOST_BM_VM(HOST_vertex_mass_spot, "../models/spot.obj");
//HOST_BM_VM(HOST_vertex_mass_sphere_400, "../models/dense_sphere_400x400.obj");
//HOST_BM_VM(HOST_vertex_mass_sphere_1000, "../models/dense_sphere_1000x1000.obj");
//HOST_BM_VM(HOST_vertex_mass_sphere_1500, "../models/dense_sphere_1500x1500.obj");
//HOST_BM_VM(HOST_vertex_mass_cube_1000, "../models/cube_1k.obj");


