#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/area.cuh"
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/host/valence.hpp"

#define DEVICE_BM_CLT(BM_NAME, FILE_NAME)                                      \
  static void BM_NAME(benchmark::State& state)                                 \
  {                                                                            \
    auto surf = TestCache::get_mesh(FILE_NAME);                                \
    auto vv = flo::host::valence(surf.faces);                                  \
    auto tv = std::accumulate(vv.begin(), vv.end(), 0);                        \
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

#define DEVICE_BM_CLM(BM_NAME, FILE_NAME)                                      \
  static void BM_NAME(benchmark::State& state)                                 \
  {                                                                            \
    auto surf = TestCache::get_mesh(FILE_NAME);                                \
    auto raw_vert_ptr = (flo::real3*)(&surf.vertices[0][0]);                   \
    auto raw_face_ptr = (int3*)(&surf.faces[0][0]);                            \
    thrust::device_vector<int3> d_faces(surf.n_faces());                       \
    thrust::copy(raw_face_ptr, raw_face_ptr + surf.n_faces(), d_faces.data()); \
    thrust::device_vector<flo::real3> d_verts(surf.n_vertices());              \
    thrust::copy(                                                              \
      raw_vert_ptr, raw_vert_ptr + surf.n_vertices(), d_verts.data());         \
    auto d_area =                                                              \
      flo::device::area(d_verts.data(), d_faces.data(), d_faces.size());       \
    thrust::device_vector<int> d_valence(surf.n_vertices());                   \
    thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1);    \
    auto d_adjacency =                                                         \
      flo::device::vertex_vertex_adjacency(d_faces.data(),                     \
                                           surf.n_faces(),                     \
                                           surf.n_vertices(),                  \
                                           d_valence.data(),                   \
                                           d_cumulative_valence.data());       \
                                                                               \
    auto d_offsets =                                                           \
      flo::device::adjacency_matrix_offset(d_faces.data(),                     \
                                           d_adjacency.data(),                 \
                                           d_cumulative_valence.data(),        \
                                           surf.n_faces());                    \
    using SparseMatrix =                                                       \
      cusp::coo_matrix<int, flo::real, cusp::device_memory>;                   \
    SparseMatrix d_L(surf.n_vertices(),                                        \
                     surf.n_vertices(),                                        \
                     d_cumulative_valence.back() + surf.n_vertices());         \
    thrust::fill(d_L.values.begin(), d_L.values.end(), 0);                     \
    for (auto _ : state)                                                       \
    {                                                                          \
      flo::device::cotangent_laplacian(d_verts.data(),                         \
                                       d_faces.data(),                         \
                                       d_area.data(),                          \
                                       d_cumulative_valence.data(),            \
                                       d_offsets.data(),                       \
                                       surf.n_faces(),                         \
                                       d_L.row_indices.data(),                 \
                                       d_L.column_indices.data(),              \
                                       d_L.values.data());                     \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_CLA(BM_NAME, FILE_NAME)                                      \
  static void BM_NAME(benchmark::State& state)                                 \
  {                                                                            \
    auto surf = TestCache::get_mesh(FILE_NAME);                                \
    auto raw_vert_ptr = (flo::real3*)(&surf.vertices[0][0]);                   \
    auto raw_face_ptr = (int3*)(&surf.faces[0][0]);                            \
    thrust::device_vector<int3> d_faces(surf.n_faces());                       \
    thrust::copy(raw_face_ptr, raw_face_ptr + surf.n_faces(), d_faces.data()); \
    thrust::device_vector<flo::real3> d_verts(surf.n_vertices());              \
    thrust::copy(                                                              \
      raw_vert_ptr, raw_vert_ptr + surf.n_vertices(), d_verts.data());         \
    auto d_area =                                                              \
      flo::device::area(d_verts.data(), d_faces.data(), d_faces.size());       \
    thrust::device_vector<int> d_valence(surf.n_vertices());                   \
    thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1);    \
    auto d_adjacency =                                                         \
      flo::device::vertex_vertex_adjacency(d_faces.data(),                     \
                                           surf.n_faces(),                     \
                                           surf.n_vertices(),                  \
                                           d_valence.data(),                   \
                                           d_cumulative_valence.data());       \
                                                                               \
    auto d_offsets =                                                           \
      flo::device::adjacency_matrix_offset(d_faces.data(),                     \
                                           d_adjacency.data(),                 \
                                           d_cumulative_valence.data(),        \
                                           surf.n_faces());                    \
    for (auto _ : state)                                                       \
    {                                                                          \
      benchmark::DoNotOptimize(                                                \
        flo::device::cotangent_laplacian(d_verts.data(),                       \
                                         d_faces.data(),                       \
                                         d_area.data(),                        \
                                         d_cumulative_valence.data(),          \
                                         d_offsets.data(),                     \
                                         surf.n_vertices(),                    \
                                         surf.n_faces(),                       \
                                         d_cumulative_valence.back()));        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_NAME)

DEVICE_BM_CLT(DEVICE_cotangent_laplacian_triplets_cube_1, "../models/cube.obj");
DEVICE_BM_CLT(DEVICE_cotangent_laplacian_triplets_spot, "../models/spot.obj");
DEVICE_BM_CLT(DEVICE_cotangent_laplacian_triplets_sphere_400,
              "../models/dense_sphere_400x400.obj");
DEVICE_BM_CLT(DEVICE_cotangent_laplacian_triplets_sphere_1000,
              "../models/dense_sphere_1000x1000.obj");
DEVICE_BM_CLM(DEVICE_cotangent_laplacian_no_alloc_cube_1, "../models/cube.obj");
DEVICE_BM_CLM(DEVICE_cotangent_laplacian_no_alloc_spot, "../models/spot.obj");
DEVICE_BM_CLM(DEVICE_cotangent_laplacian_no_alloc_sphere_400,
              "../models/dense_sphere_400x400.obj");
DEVICE_BM_CLM(DEVICE_cotangent_laplacian_no_alloc_sphere_1000,
              "../models/dense_sphere_1000x1000.obj");
DEVICE_BM_CLA(DEVICE_cotangent_laplacian_atomic_cube_1, "../models/cube.obj");
DEVICE_BM_CLA(DEVICE_cotangent_laplacian_atomic_spot, "../models/spot.obj");
DEVICE_BM_CLA(DEVICE_cotangent_laplacian_atomic_sphere_400,
              "../models/dense_sphere_400x400.obj");
DEVICE_BM_CLA(DEVICE_cotangent_laplacian_atomic_sphere_1000,
              "../models/dense_sphere_1000x1000.obj");
// DEVICE_BM_CL(DEVICE_face_area_sphere_1500,
// "../models/dense_sphere_1500x1500.obj");
// DEVICE_BM_CL(DEVICE_face_area_cube_1000,
// "../models/cube_1k.obj");
