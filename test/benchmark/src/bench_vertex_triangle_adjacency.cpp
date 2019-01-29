#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/vertex_triangle_adjacency.hpp"
#include "flo/load_mesh.hpp"

#define HOST_BM_VTA(BM_NAME, FILE_NAME) \
static void BM_NAME(benchmark::State& state) \
{ \
  auto surf = flo::load_mesh(FILE_NAME); \
  std::vector<int> adjacency(surf.n_faces() * 3); \
  std::vector<int> valence(surf.n_vertices()); \
  std::vector<int> cumulative_valence(surf.n_vertices() + 1); \
  for (auto _ : state) \
  { \
    flo::host::vertex_triangle_adjacency( \
        surf.faces, surf.n_vertices(), adjacency, valence, cumulative_valence);\
  }\
}\
BENCHMARK(BM_NAME)

HOST_BM_VTA(HOST_vertex_triangle_adjacency_cube, "../models/cube.obj");
HOST_BM_VTA(HOST_vertex_triangle_adjacency_spot, "../models/spot.obj");
HOST_BM_VTA(HOST_vertex_triangle_adjacency_sphere_400, "../models/dense_sphere_400x400.obj");
//HOST_BM_VTA(HOST_vertex_triangle_adjacency_sphere_1000, "../models/dense_sphere_1000x1000.obj");
//HOST_BM_VTA(HOST_vertex_triangle_adjacency_sphere_1500, "../models/dense_sphere_1500x1500.obj");
//HOST_BM_VTA(HOST_vertex_triangle_adjacency_cube_1000, "../models/cube_1k.obj");


