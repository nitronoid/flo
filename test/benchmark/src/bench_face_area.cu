#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/area.cuh"
#include "flo/load_mesh.hpp"

#define DEVICE_BM_FA(BM_NAME, FILE_NAME) \
static void BM_NAME(benchmark::State& state) \
{ \
  auto surf = flo::load_mesh(FILE_NAME); \
  thrust::device_vector<Eigen::Vector3i> d_faces = surf.faces;\
  thrust::device_vector<Eigen::Vector3d> d_verts = surf.vertices;\
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
