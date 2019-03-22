#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/area.cuh"
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/host/valence.hpp"

namespace
{
static void bench_impl(std::string name, benchmark::State& state)
{
  // Load our surface from the cache
  auto surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Obtain the face areas
  cusp::array1d<flo::real, cusp::device_memory> d_area(surf.n_faces());
  flo::device::area(surf.vertices, surf.faces, d_area);
  // Obtain the vertex vertex adjacency and valence
  cusp::array1d<int, cusp::device_memory> d_adjacency(surf.n_faces() * 12);
  cusp::array1d<int, cusp::device_memory> d_valence(surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
    surf.n_vertices() + 1);
  int n_adjacency = flo::device::vertex_vertex_adjacency(
    surf.faces,
    d_adjacency,
    d_valence,
    {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()});
  d_adjacency.resize(n_adjacency);

  // Obtain the address offsets to write our matrix entries
  cusp::array1d<int2, cusp::device_memory> d_offsets(surf.n_faces() * 3);
  flo::device::adjacency_matrix_offset(
    surf.faces, d_adjacency, d_cumulative_valence, d_offsets);

  using SparseMatrix = cusp::coo_matrix<int, flo::real, cusp::device_memory>;
  SparseMatrix d_L(surf.n_vertices(),
                   surf.n_vertices(),
                   d_cumulative_valence.back() + surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_diagonals(surf.n_vertices());
  for (auto _ : state)
  {
    flo::device::cotangent_laplacian(
      surf.vertices, surf.faces, d_area, d_offsets, d_diagonals, d_L);
  }
}
}  // namespace

#define FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_cotangent_laplacian_##NAME(benchmark::State& state) \
  {                                                                      \
    bench_impl(#NAME, state);                                             \
  }                                                                      \
  BENCHMARK(DEVICE_cotangent_laplacian_##NAME);

FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(cube)
FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(spot)
FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(dense_sphere_400x400)
FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK(dense_sphere_1000x1000)

#undef FLO_COTANGENT_LAPLACIAN_DEVICE_BENCHMARK
