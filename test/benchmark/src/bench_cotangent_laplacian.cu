#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/area.cuh"
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/load_mesh.hpp"
#include "flo/host/valence.hpp"

#define DEVICE_BM_CLK(BM_NAME, FILE_NAME)                                      \
  static void BM_NAME(benchmark::State& state)                                 \
  {                                                                            \
    auto surf = flo::load_mesh(FILE_NAME);                                     \
    auto raw_vert_ptr = (flo::real3*)(&surf.vertices[0][0]);                   \
    auto raw_face_ptr = (int3*)(&surf.faces[0][0]);                            \
    thrust::device_vector<int3> d_faces(surf.n_faces());                       \
    thrust::copy(raw_face_ptr, raw_face_ptr + surf.n_faces(), d_faces.data()); \
    thrust::device_vector<flo::real3> d_verts(surf.n_vertices());              \
    thrust::copy(                                                              \
      raw_vert_ptr, raw_vert_ptr + surf.n_vertices(), d_verts.data());         \
    auto d_area =                                                              \
      flo::device::area(d_verts.data(), d_faces.data(), d_faces.size());       \
    const int ntriplets = surf.n_faces() * 12;                                 \
    thrust::device_vector<int> I(ntriplets);                                   \
    thrust::device_vector<int> J(ntriplets);                                   \
    thrust::device_vector<flo::real> V(ntriplets);                             \
    dim3 block_dim;                                                            \
    block_dim.x = 4;                                                           \
    block_dim.y = 3;                                                           \
    block_dim.z = 64;                                                          \
    size_t nthreads_per_block = block_dim.x * block_dim.y * block_dim.z;       \
    size_t nblocks = ntriplets / nthreads_per_block + 1;                       \
    size_t shared_memory_size = sizeof(flo::real) * block_dim.z * 15;          \
    for (auto _ : state)                                                       \
    {                                                                          \
      flo::device::d_cotangent_laplacian_triplets<<<nblocks,                   \
                                                    block_dim,                 \
                                                    shared_memory_size>>>(     \
        d_verts.data(),                                                        \
        d_faces.data(),                                                        \
        d_area.data(),                                                         \
        surf.n_faces(),                                                        \
        I.data(),                                                              \
        J.data(),                                                              \
        V.data());                                                             \
      cudaDeviceSynchronize();                                                 \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_CL(BM_NAME, FILE_NAME)                                       \
  static void BM_NAME(benchmark::State& state)                                 \
  {                                                                            \
    auto surf = flo::load_mesh(FILE_NAME);                                     \
    auto vv = flo::host::valence(surf.faces); \
    auto tv = std::accumulate(vv.begin(), vv.end(), 0);\
    auto raw_vert_ptr = (flo::real3*)(&surf.vertices[0][0]);                   \
    auto raw_face_ptr = (int3*)(&surf.faces[0][0]);                            \
    thrust::device_vector<int3> d_faces(surf.n_faces());                       \
    thrust::copy(raw_face_ptr, raw_face_ptr + surf.n_faces(), d_faces.data()); \
    thrust::device_vector<flo::real3> d_verts(surf.n_vertices());              \
    thrust::copy(                                                              \
      raw_vert_ptr, raw_vert_ptr + surf.n_vertices(), d_verts.data());         \
    auto d_area =                                                              \
      flo::device::area(d_verts.data(), d_faces.data(), d_faces.size());       \
    for (auto _ : state)                                                       \
    {                                                                          \
      benchmark::DoNotOptimize(                                                \
        flo::device::cotangent_laplacian(d_verts.data(),                       \
                                         d_faces.data(),                       \
                                         d_area.data(),                        \
                                         surf.n_vertices(),                    \
                                         surf.n_faces(),                       \
                                         tv));                                 \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_NAME)

DEVICE_BM_CL(DEVICE_cotangent_laplacian_cube_1, "../models/cube.obj");
DEVICE_BM_CL(DEVICE_cotangent_laplacian_spot, "../models/spot.obj");
DEVICE_BM_CL(DEVICE_cotangent_laplacian_sphere_400,
             "../models/dense_sphere_400x400.obj");
DEVICE_BM_CL(DEVICE_cotangent_laplacian_sphere_1000,
             "../models/dense_sphere_1000x1000.obj");
DEVICE_BM_CLK(DEVICE_cotangent_laplacian_kernel_cube_1, "../models/cube.obj");
DEVICE_BM_CLK(DEVICE_cotangent_laplacian_kernel_spot, "../models/spot.obj");
DEVICE_BM_CLK(DEVICE_cotangent_laplacian_kernel_sphere_400,
              "../models/dense_sphere_400x400.obj");
DEVICE_BM_CLK(DEVICE_cotangent_laplacian_kernel_sphere_1000,
              "../models/dense_sphere_1000x1000.obj");
// DEVICE_BM_CL(DEVICE_face_area_sphere_1500,
// "../models/dense_sphere_1500x1500.obj");
// DEVICE_BM_CL(DEVICE_face_area_cube_1000,
// "../models/cube_1k.obj");
