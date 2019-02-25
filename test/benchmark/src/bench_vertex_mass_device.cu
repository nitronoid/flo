#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/vertex_mass.cuh"

#define DEVICE_BM_VM(BM_NAME, FILE_NAME)                                    \
  static void BM_NAME(benchmark::State& state)                              \
  {                                                                         \
    auto surf = TestCache::get_mesh<TestCache::DEVICE>(FILE_NAME);          \
    auto faces_copy = surf.faces;                                           \
    thrust::device_vector<int> d_adjacency(surf.n_faces() * 3);             \
    thrust::device_vector<int> d_valence(surf.n_vertices());                \
    thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1); \
    flo::device::vertex_triangle_adjacency(                                 \
      thrust::device_ptr<int>{(int*)faces_copy.data().get()},               \
      surf.n_faces(),                                                       \
      surf.n_vertices(),                                                    \
      d_adjacency.data(),                                                   \
      d_valence.data(),                                                     \
      d_cumulative_valence.data());                                         \
    std::vector<flo::real> h_area(12, 0.5);                                 \
    thrust::device_vector<flo::real> d_area = h_area;                       \
    for (auto _ : state)                                                    \
    {                                                                       \
      benchmark::DoNotOptimize(                                             \
        flo::device::vertex_mass(d_area.data(),                             \
                                 d_adjacency.data(),                        \
                                 d_valence.data(),                          \
                                 d_cumulative_valence.data(),               \
                                 d_area.size(),                             \
                                 d_valence.size()));                        \
    }                                                                       \
  }                                                                         \
  BENCHMARK(BM_NAME)

DEVICE_BM_VM(DEVICE_vertex_mass_cube_1, "../models/cube.obj");
DEVICE_BM_VM(DEVICE_vertex_mass_spot, "../models/spot.obj");
DEVICE_BM_VM(DEVICE_vertex_mass_sphere_400,
             "../models/dense_sphere_400x400.obj");
DEVICE_BM_VM(DEVICE_vertex_mass_sphere_1000,
             "../models/dense_sphere_1000x1000.obj");
// DEVICE_BM_VM(DEVICE_vertex_mass_sphere_1500,
// "../models/dense_sphere_1500x1500.obj");
// DEVICE_BM_VM(DEVICE_vertex_mass_cube_1000, "../models/cube_1k.obj");
