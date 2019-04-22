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
using SparseDeviceMatrixQ =
  cusp::coo_matrix<int, flo::real4, cusp::device_memory>;
using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;

void bench_impl(std::string name, benchmark::State& state)
{
  // Load our surface from the cache
  auto surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Arbitrary constant rho
  cusp::array1d<flo::real, cusp::device_memory> d_rho(surf.n_vertices(), 3.f);
  // Obtain the face areas
  cusp::array1d<flo::real, cusp::device_memory> d_area(surf.n_faces());
  flo::device::face_area(surf.vertices, surf.faces, d_area);
  // Obtain the vertex vertex adjacency and valence
  cusp::array1d<int, cusp::device_memory> d_adjacency(surf.n_faces() * 6);
  cusp::array1d<int, cusp::device_memory> d_adjacency_keys(surf.n_faces() * 6);
  cusp::array1d<int, cusp::device_memory> d_valence(surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
    surf.n_vertices() + 1);
  int n_adjacency = flo::device::vertex_vertex_adjacency(
    surf.faces,
    d_adjacency_keys,
    d_adjacency,
    d_valence,
    {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()});
  const auto& ad = d_adjacency.subarray(0, n_adjacency);

  //// Obtain the address offsets to write our matrix entries
  cusp::array2d<int, cusp::device_memory> d_offsets(6, surf.n_faces());
  flo::device::adjacency_matrix_offset(
    surf.faces, d_adjacency, d_cumulative_valence, d_offsets);

  // Obtain the vertex triangle adjacency and valence
  cusp::array1d<int, cusp::device_memory> d_triangle_adjacency(surf.n_faces() *
                                                               3);
  cusp::array1d<int, cusp::device_memory> d_triangle_valence(surf.n_vertices());
  cusp::array1d<int, cusp::device_memory> d_triangle_cumulative_valence(
    surf.n_vertices() + 1);
  thrust::device_vector<int> temp(surf.n_faces() * 3);
  // Run the function
  flo::device::vertex_triangle_adjacency(
    surf.faces,
    temp.data(),
    d_triangle_adjacency,
    d_triangle_valence,
    {d_triangle_cumulative_valence.begin() + 1,
     d_triangle_cumulative_valence.end()});

  cusp::array1d<int, cusp::device_memory> d_triangle_adjacency_keys(
    d_triangle_cumulative_valence.back());
  thrust::copy_n(thrust::constant_iterator<int>(1),
                 surf.n_vertices() - 1,
                 thrust::make_permutation_iterator(
                   d_triangle_adjacency_keys.begin(),
                   d_triangle_cumulative_valence.begin() + 1));
  thrust::inclusive_scan(d_triangle_adjacency_keys.begin(),
                         d_triangle_adjacency_keys.end(),
                         d_triangle_adjacency_keys.begin());

  // Allocate a sparse quaternion matrix to store our result
  SparseDeviceMatrixQ d_Dq(surf.n_vertices(),
                           surf.n_vertices(),
                           d_cumulative_valence.back() + surf.n_vertices());

  // Allocate a dense 1 dimensional array to receive diagonal element indices
  cusp::array1d<int, cusp::device_memory> d_diagonals(surf.n_vertices());

  //// Allocate our real matrix for solving
  // SparseDeviceMatrix d_Dr(
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
FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK(dense_sphere_400x400)
// FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK(dense_sphere_1000x1000)

#undef FLO_INTRINSIC_DIRAC_DEVICE_BENCHMARK
