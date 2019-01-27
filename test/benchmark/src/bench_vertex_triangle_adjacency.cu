#include "test_common.h"
#include "flo/host/vertex_triangle_adjacency.hpp"
#include "flo/device/vertex_triangle_adjacency.cuh"

extern const char k_cube[] = "../models/cube.obj";
using CubeHostFixture = SurfaceHostFixture<k_cube>;
using CubeDeviceFixture = SurfaceDeviceFixture<k_cube>;

extern const char k_spot[] = "../models/spot.obj";
using SpotHostFixture = SurfaceHostFixture<k_spot>;
using SpotDeviceFixture = SurfaceDeviceFixture<k_spot>;

#define RUN_HOST_VTA(NAME) \
BENCHMARK_DEFINE_F(NAME, vertex_triangle_adjacency)(benchmark::State& state)\
{\
  for (auto _ : state)\
  {\
    flo::host::vertex_triangle_adjacency(\
        surf.faces, surf.n_vertices(), adjacency, valence, cumulative_valence);\
  }\
}\
BENCHMARK_REGISTER_F(NAME, vertex_triangle_adjacency)

#define RUN_DEVICE_VTA( NAME ) \
BENCHMARK_DEFINE_F( NAME , vertex_triangle_adjacency)(benchmark::State& state)\
{\
  for (auto _ : state)\
  {\
    flo::device::vertex_triangle_adjacency(\
        d_face_verts.data(), \
        surf.n_faces(),\
        surf.n_vertices(), \
        d_adjacency.data(), \
        d_valence.data(), \
        d_cumulative_valence.data());\
  }\
}\
BENCHMARK_REGISTER_F( NAME , vertex_triangle_adjacency)


// CUBE host benchmark
RUN_HOST_VTA(CubeHostFixture);
// CUBE device benchmark
RUN_DEVICE_VTA(CubeDeviceFixture);
// SPOT host benchmark
RUN_HOST_VTA(SpotHostFixture);
// SPOT device benchmark
RUN_DEVICE_VTA(SpotDeviceFixture);
