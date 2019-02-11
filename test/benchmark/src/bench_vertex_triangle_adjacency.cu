#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/vertex_triangle_adjacency.cuh"

#define DEVICE_BM_VTA(BM_NAME, FILE_NAME) \
static void BM_NAME(benchmark::State& state) \
{ \
  auto surf = TestCache::get_mesh(FILE_NAME);                                \
  thrust::device_vector<int> d_face_verts(surf.n_faces() * 3); \
  thrust::copy_n((&surf.faces[0][0]), surf.n_faces() * 3, d_face_verts.data());\
  thrust::device_vector<int> d_adjacency(surf.n_faces() * 3);\
  thrust::device_vector<int> d_valence(surf.n_vertices());\
  thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1);\
  for (auto _ : state)\
  {\
    flo::device::vertex_triangle_adjacency(\
        d_face_verts.data(), \
        surf.n_faces(),\
        surf.n_vertices(), \
        d_adjacency.data(),\
        d_valence.data(),\
        d_cumulative_valence.data());\
  }\
}\
BENCHMARK(BM_NAME)

DEVICE_BM_VTA(DEVICE_vertex_triangle_adjacency_cube_1, "../models/cube.obj");
DEVICE_BM_VTA(DEVICE_vertex_triangle_adjacency_spot, "../models/spot.obj");
DEVICE_BM_VTA(DEVICE_vertex_triangle_adjacency_sphere_400, "../models/dense_sphere_400x400.obj");
//DEVICE_BM_VTA(DEVICE_vertex_triangle_adjacency_sphere_1000, "../models/dense_sphere_1000x1000.obj");
//DEVICE_BM_VTA(DEVICE_vertex_triangle_adjacency_sphere_1500, "../models/dense_sphere_1500x1500.obj");
//DEVICE_BM_VTA(DEVICE_vertex_triangle_adjacency_cube_1000, "../models/cube_1k.obj");
