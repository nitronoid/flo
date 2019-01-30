#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/area.cuh"
#include "flo/load_mesh.hpp"

#define DEVICE_BM_FA(BM_NAME, FILE_NAME) \
static void BM_NAME(benchmark::State& state) \
{ \
  auto surf = flo::load_mesh(FILE_NAME); \
  auto raw_vert_ptr = (double3*)(&surf.vertices[0][0]);\
  auto raw_face_ptr = (int3*)(&surf.faces[0][0]);\
  thrust::device_vector<int3> d_faces(surf.n_faces());\
  thrust::copy(raw_face_ptr, raw_face_ptr + surf.n_faces(), d_faces.data());\
  thrust::device_vector<double3> d_verts(surf.n_vertices());\
  thrust::copy(raw_vert_ptr, raw_vert_ptr + surf.n_vertices(), d_verts.data());\
  for (auto _ : state)\
  {\
    benchmark::DoNotOptimize(\
        flo::device::area(d_verts.data(), d_faces.data(), d_faces.size()));\
  }\
}\
BENCHMARK(BM_NAME)

DEVICE_BM_FA(DEVICE_face_area_cube_1, "../models/cube.obj");
DEVICE_BM_FA(DEVICE_face_area_spot, "../models/spot.obj");
DEVICE_BM_FA(DEVICE_face_area_sphere_400, "../models/dense_sphere_400x400.obj");
//DEVICE_BM_FA(DEVICE_face_area_sphere_1000, "../models/dense_sphere_1000x1000.obj");
//DEVICE_BM_FA(DEVICE_face_area_sphere_1500, "../models/dense_sphere_1500x1500.obj");
//DEVICE_BM_FA(DEVICE_face_area_cube_1000, "../models/cube_1k.obj");
