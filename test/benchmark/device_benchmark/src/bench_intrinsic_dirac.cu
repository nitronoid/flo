#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/face_area.cuh"
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include <cusp/coo_matrix.h>

namespace
{
void bench_impl(std::string name, benchmark::State& state)
{
  // Set-up matrix path
  const std::string mp = "../../matrices/" + name;
  // Load our surface from the cache
  auto surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Arbitrary constant rho
  DeviceVectorR d_rho(surf.n_vertices(), 3.f);

  // Read all our dependencies from disk
  auto d_area = read_device_vector<flo::real>(mp + "/face_area/face_area.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_adjacency_keys =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto d_adjacency =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto d_triangle_adjacency_keys = read_device_vector<int>(
    mp + "/vertex_triangle_adjacency/adjacency_keys.mtx");
  auto d_triangle_adjacency =
    read_device_vector<int>(mp + "/vertex_triangle_adjacency/adjacency.mtx");
  auto d_offsets =
    read_device_dense_matrix<int>(mp + "/adjacency_matrix_offset/offsets.mtx");

  // Allocate a sparse quaternion matrix to store our result
  DeviceSparseMatrixQ d_Dq(surf.n_vertices(),
                           surf.n_vertices(),
                           d_cumulative_valence.back() + surf.n_vertices());

  // Allocate a dense 1 dimensional array to receive diagonal element indices
  DeviceVectorI d_diagonals(surf.n_vertices());

  //// Allocate our real matrix for solving
  // DeviceSparseMatrixR d_Dr(
  //  surf.n_vertices() * 4, surf.n_vertices() * 4, d_Dq.values.size() * 16);

  for (auto _ : state)
  {
    // Run our function
    flo::device::intrinsic_dirac(surf.vertices,
                                 surf.faces,
                                 d_area,
                                 d_rho,
                                 d_offsets,
                                 d_adjacency_keys,
                                 d_adjacency,
                                 d_cumulative_valence,
                                 d_triangle_adjacency_keys,
                                 d_triangle_adjacency,
                                 d_diagonals,
                                 d_Dq);

    //// Transform our quaternion matrix to a real matrix
    // flo::device::to_real_quaternion_matrix(
    //  d_Dq,
    //  {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()},
    //  d_Dr);
  }
}
}  // namespace

#define FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_intrinsic_dirac_##NAME(benchmark::State& state) \
  {                                                                  \
    bench_impl(#NAME, state);                                        \
  }                                                                  \
  BENCHMARK(DEVICE_intrinsic_dirac_##NAME);

FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK(cube)
FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK(spot)
FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK(bunny)

#undef FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK
