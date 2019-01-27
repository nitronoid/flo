#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/vertex_triangle_adjacency.hpp"
#include "flo/device/vertex_triangle_adjacency.cuh"
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

#define DEVICE_BM_VTA(BM_NAME, FILE_NAME) \
static void BM_NAME(benchmark::State& state) \
{ \
  auto surf = flo::load_mesh(FILE_NAME); \
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

HOST_BM_VTA(BM_cube_host_vertex_triangle_adjacency, "../models/cube.obj");
DEVICE_BM_VTA(BM_cube_device_vertex_triangle_adjacency, "../models/cube.obj");

HOST_BM_VTA(BM_spot_host_vertex_triangle_adjacency, "../models/spot.obj");
DEVICE_BM_VTA(BM_spot_device_vertex_triangle_adjacency, "../models/spot.obj");

HOST_BM_VTA(BM_dense_sphere_400_host_vertex_triangle_adjacency, "../models/dense_sphere_400x400.obj");
DEVICE_BM_VTA(BM_dense_sphere_400_device_vertex_triangle_adjacency, "../models/dense_sphere_400x400.obj");

HOST_BM_VTA(BM_dense_sphere_1k_host_vertex_triangle_adjacency, "../models/dense_sphere_1000x1000.obj");
DEVICE_BM_VTA(BM_dense_sphere_1k_device_vertex_triangle_adjacency, "../models/dense_sphere_1000x1000.obj");

