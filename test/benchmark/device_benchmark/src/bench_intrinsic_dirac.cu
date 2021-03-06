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

  // Read all our dependencies from disk
  auto d_rho = read_device_vector<flo::real>(mp + "/project_basis/rho.mtx");
  auto d_area = read_device_vector<flo::real>(mp + "/face_area/face_area.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_triangle_adjacency_keys = read_device_vector<int>(
    mp + "/vertex_triangle_adjacency/adjacency_keys.mtx");
  auto d_triangle_adjacency =
    read_device_vector<int>(mp + "/vertex_triangle_adjacency/adjacency.mtx");
  auto d_entry_indices =
    read_device_dense_matrix<int>(mp + "/adjacency_matrix_indices/indices.mtx");
  auto d_diagonal_indices =
    read_device_vector<int>(mp + "/cotangent_laplacian/diagonals.mtx");
  auto d_D = 
    read_device_sparse_matrix<flo::real>(mp + "/intrinsic_dirac/intrinsic_dirac.mtx");


  // Add an ascending sequence to the cumulative valence to account for
  // diagonals
  thrust::transform(d_cumulative_valence.begin() + 1,
                    d_cumulative_valence.end(),
                    thrust::make_counting_iterator(1),
                    d_cumulative_valence.begin() + 1,
                    thrust::plus<int>());

  // Allocate a sparse quaternion matrix to store our result
  DeviceSparseMatrixQ d_Dq(surf.n_vertices(),
                           surf.n_vertices(),
                           d_cumulative_valence.back() + surf.n_vertices());
  d_Dq.row_indices = d_D.row_indices;
  d_Dq.column_indices = d_D.column_indices;

  // Allocate our real matrix for solving
  DeviceSparseMatrixR d_Dr(
    surf.n_vertices() * 4, surf.n_vertices() * 4, d_Dq.values.size() * 16);

  for (auto _ : state)
  {
    // Run our function
    flo::device::intrinsic_dirac_values(surf.vertices,
                                        surf.faces,
                                        d_area,
                                        d_rho,
                                        d_triangle_adjacency_keys,
                                        d_triangle_adjacency,
                                        d_entry_indices,
                                        d_diagonal_indices,
                                        d_Dq);

    // Transform our quaternion matrix to a real matrix
    flo::device::to_quaternion_matrix(
      d_Dq,
      {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()},
      d_Dr);
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
FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK(tyra)

#undef FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK
