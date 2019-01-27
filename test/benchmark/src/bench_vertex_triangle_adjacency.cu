#include <benchmark/benchmark.h>
#include "test_common.h"
#include "flo/host/vertex_triangle_adjacency.hpp"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/load_mesh.hpp"

static void BM_cube_host_vertex_triangle_adjacency(benchmark::State& state) 
{
  auto surf = flo::load_mesh("../models/cube.obj");
  std::vector<int> adjacency(surf.n_faces() * 3);
  std::vector<int> valence(surf.n_vertices());
  std::vector<int> cumulative_valence(surf.n_vertices() + 1);
  for (auto _ : state)
  {
    flo::host::vertex_triangle_adjacency(
        surf.faces, surf.n_vertices(), adjacency, valence, cumulative_valence);
  }
}
BENCHMARK(BM_cube_host_vertex_triangle_adjacency);

static void BM_cube_device_vertex_triangle_adjacency(benchmark::State& state) 
{
  auto surf = flo::load_mesh("../models/cube.obj");
  thrust::device_vector<int> d_face_verts(surf.n_faces() * 3);
  thrust::copy_n((&surf.faces[0][0]), surf.n_faces() * 3, d_face_verts.data());
  // Declare device side arrays to dump our results
  thrust::device_vector<int> d_adjacency(surf.n_faces() * 3);
  thrust::device_vector<int> d_valence(surf.n_vertices());
  thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1);
  for (auto _ : state)
  {
    flo::device::vertex_triangle_adjacency(
        d_face_verts.data(), 
        surf.n_faces(),
        surf.n_vertices(), 
        d_adjacency.data(),
        d_valence.data(),
        d_cumulative_valence.data());
  }
}
BENCHMARK(BM_cube_device_vertex_triangle_adjacency);

static void BM_spot_host_vertex_triangle_adjacency(benchmark::State& state) 
{
  auto surf = flo::load_mesh("../models/spot.obj");
  std::vector<int> adjacency(surf.n_faces() * 3);
  std::vector<int> valence(surf.n_vertices());
  std::vector<int> cumulative_valence(surf.n_vertices() + 1);
  for (auto _ : state)
  {
    flo::host::vertex_triangle_adjacency(
        surf.faces, surf.n_vertices(), adjacency, valence, cumulative_valence);
  }
}
BENCHMARK(BM_spot_host_vertex_triangle_adjacency);

static void BM_spot_device_vertex_triangle_adjacency(benchmark::State& state) 
{
  auto surf = flo::load_mesh("../models/spot.obj");
  thrust::device_vector<int> d_face_verts(surf.n_faces() * 3);
  thrust::copy_n((&surf.faces[0][0]), surf.n_faces() * 3, d_face_verts.data());
  // Declare device side arrays to dump our results
  thrust::device_vector<int> d_adjacency(surf.n_faces() * 3);
  thrust::device_vector<int> d_valence(surf.n_vertices());
  thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1);
  for (auto _ : state)
  {
    flo::device::vertex_triangle_adjacency(
        d_face_verts.data(), 
        surf.n_faces(),
        surf.n_vertices(), 
        d_adjacency.data(),
        d_valence.data(),
        d_cumulative_valence.data());
  }
}
BENCHMARK(BM_spot_device_vertex_triangle_adjacency);

static void BM_dense_sphere_host_vertex_triangle_adjacency(benchmark::State& state) 
{
  auto surf = flo::load_mesh("../models/dense_sphere_1000x1000.obj");
  std::vector<int> adjacency(surf.n_faces() * 3);
  std::vector<int> valence(surf.n_vertices());
  std::vector<int> cumulative_valence(surf.n_vertices() + 1);
  for (auto _ : state)
  {
    flo::host::vertex_triangle_adjacency(
        surf.faces, surf.n_vertices(), adjacency, valence, cumulative_valence);
  }
}
BENCHMARK(BM_dense_sphere_host_vertex_triangle_adjacency);

static void BM_dense_sphere_device_vertex_triangle_adjacency(benchmark::State& state) 
{
  auto surf = flo::load_mesh("../models/dense_sphere_1000x1000.obj");
  thrust::device_vector<int> d_face_verts(surf.n_faces() * 3);
  thrust::copy_n((&surf.faces[0][0]), surf.n_faces() * 3, d_face_verts.data());
  // Declare device side arrays to dump our results
  thrust::device_vector<int> d_adjacency(surf.n_faces() * 3);
  thrust::device_vector<int> d_valence(surf.n_vertices());
  thrust::device_vector<int> d_cumulative_valence(surf.n_vertices() + 1);
  for (auto _ : state)
  {
    flo::device::vertex_triangle_adjacency(
        d_face_verts.data(), 
        surf.n_faces(),
        surf.n_vertices(), 
        d_adjacency.data(),
        d_valence.data(),
        d_cumulative_valence.data());
  }
}
BENCHMARK(BM_dense_sphere_device_vertex_triangle_adjacency);
