#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/area.cuh"
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/host/valence.hpp"

#define DEVICE_BM_CLT(BM_NAME, FILE_NAME)                                   \
  static void BM_NAME(benchmark::State& state)                              \
  {                                                                         \
    auto surf = TestCache::get_mesh<TestCache::DEVICE>(FILE_NAME);          \
    auto d_area = flo::device::area(                                        \
      surf.vertices.data(), surf.faces.data(), surf.faces.size());          \
    thrust::device_vector<int> d_valence(surf.n_vertices());                \
    thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1); \
    auto d_adjacency =                                                      \
      flo::device::vertex_vertex_adjacency(surf.faces.data(),               \
                                           surf.n_faces(),                  \
                                           surf.n_vertices(),               \
                                           d_valence.data(),                \
                                           d_cumulative_valence.data());    \
    for (auto _ : state)                                                    \
    {                                                                       \
      benchmark::DoNotOptimize(                                             \
        flo::device::cotangent_laplacian(surf.vertices.data(),              \
                                         surf.faces.data(),                 \
                                         d_area.data(),                     \
                                         surf.n_vertices(),                 \
                                         surf.n_faces(),                    \
                                         d_cumulative_valence.back()));     \
    }                                                                       \
  }                                                                         \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_CLA(BM_NAME, FILE_NAME)                                   \
  static void BM_NAME(benchmark::State& state)                              \
  {                                                                         \
    auto surf = TestCache::get_mesh<TestCache::DEVICE>(FILE_NAME);          \
    auto d_area = flo::device::area(                                        \
      surf.vertices.data(), surf.faces.data(), surf.faces.size());          \
    thrust::device_vector<int> d_valence(surf.n_vertices());                \
    thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1); \
    auto d_adjacency =                                                      \
      flo::device::vertex_vertex_adjacency(surf.faces.data(),               \
                                           surf.n_faces(),                  \
                                           surf.n_vertices(),               \
                                           d_valence.data(),                \
                                           d_cumulative_valence.data());    \
                                                                            \
    auto d_offsets =                                                        \
      flo::device::adjacency_matrix_offset(surf.faces.data(),               \
                                           d_adjacency.data(),              \
                                           d_cumulative_valence.data(),     \
                                           surf.n_faces());                 \
    using SparseMatrix =                                                    \
      cusp::coo_matrix<int, flo::real, cusp::device_memory>;                \
    SparseMatrix d_L(surf.n_vertices(),                                     \
                     surf.n_vertices(),                                     \
                     d_cumulative_valence.back() + surf.n_vertices());      \
    thrust::device_vector<int> d_diagonals(surf.n_vertices());              \
    for (auto _ : state)                                                    \
    {                                                                       \
      flo::device::cotangent_laplacian(surf.vertices.data(),                \
                                       surf.faces.data(),                   \
                                       d_area.data(),                       \
                                       d_cumulative_valence.data(),         \
                                       d_offsets.data(),                    \
                                       surf.n_vertices(),                   \
                                       surf.n_faces(),                      \
                                       d_cumulative_valence.back(),         \
                                       d_diagonals.data(),                  \
                                       d_L.row_indices.data(),              \
                                       d_L.column_indices.data(),           \
                                       d_L.values.data());                  \
    }                                                                       \
  }                                                                         \
  BENCHMARK(BM_NAME)

DEVICE_BM_CLA(DEVICE_cotangent_laplacian_cube_1, "cube.obj");
DEVICE_BM_CLA(DEVICE_cotangent_laplacian_spot, "spot.obj");
DEVICE_BM_CLA(DEVICE_cotangent_laplacian_sphere_400, 
    "dense_sphere_400x400.obj");
DEVICE_BM_CLA(DEVICE_cotangent_laplacian_sphere_1000, 
    "dense_sphere_1000x1000.obj");
