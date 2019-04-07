#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/vertex_mass.cuh"
#include "flo/device/face_area.cuh"

#include "flo/host/cotangent_laplacian.hpp"

namespace
{
using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;

void test(std::string name)
{
  // Set-up matrix path
  const std::string matrix_prefix = "../matrices/" + name;
  // Load the surface from our mesh cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read all our dependencies from disk
  cusp::array1d<int, cusp::host_memory> int_temp;
  cusp::array1d<flo::real, cusp::host_memory> real_temp;
  cusp::io::read_matrix_market_file(real_temp,
                                    matrix_prefix + "/area/area.mtx");
  cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;
  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_valence = int_temp;
  cusp::io::read_matrix_market_file(
    int_temp,
    matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;
  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/vertex_vertex_adjacency/adjacency.mtx");
  cusp::array1d<int, cusp::device_memory> d_adjacency = int_temp;

  cusp::array2d<int, cusp::device_memory> d_offsets(6, surf.n_faces());
  // Run the function
  flo::device::adjacency_matrix_offset(
    surf.faces, d_adjacency, d_cumulative_valence, d_offsets);

  cusp::array1d<int, cusp::device_memory> d_adjacency_keys(
    d_cumulative_valence.back());
  thrust::copy_n(thrust::constant_iterator<int>(1),
                 surf.n_vertices() - 1,
                 thrust::make_permutation_iterator(
                   d_adjacency_keys.begin(), d_cumulative_valence.begin() + 1));
  thrust::inclusive_scan(
    d_adjacency_keys.begin(), d_adjacency_keys.end(), d_adjacency_keys.begin());

  // Allocate a sparse matrix to store our result
  SparseDeviceMatrix d_L(surf.n_vertices(),
                         surf.n_vertices(),
                         d_cumulative_valence.back() + surf.n_vertices());

  // Allocate a dense 1 dimensional array to receive diagonal element indices
  cusp::array1d<int, cusp::device_memory> d_diagonals(surf.n_vertices());
  // Run our function
  flo::device::cotangent_laplacian(surf.vertices,
                                   surf.faces,
                                   d_area,
                                   d_offsets,
                                   d_adjacency_keys,
                                   d_adjacency,
                                   d_cumulative_valence,
                                   d_diagonals,
                                   d_L);

  // Copy our results back to the host side
  SparseHostMatrix h_L;
  h_L.column_indices = d_L.column_indices;
  h_L.row_indices = d_L.row_indices;
  h_L.values = d_L.values;

  // Load our expected results from disk
  cusp::array1d<int, cusp::host_memory> h_diagonals = d_diagonals;
  SparseHostMatrix expected_L;
  cusp::io::read_matrix_market_file(
    expected_L, matrix_prefix + "/cotangent_laplacian/cotangent_laplacian.mtx");
  cusp::array1d<int, cusp::host_memory> expected_diagonals;
  cusp::io::read_matrix_market_file(
    expected_diagonals, matrix_prefix + "/cotangent_laplacian/diagonals.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_diagonals, Pointwise(Eq(), expected_diagonals));
  EXPECT_THAT(h_L.row_indices, Pointwise(Eq(), expected_L.row_indices));
  EXPECT_THAT(h_L.column_indices, Pointwise(Eq(), expected_L.column_indices));
  EXPECT_THAT(h_L.values,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_L.values));
}
}  // namespace

#define FLO_COTANGENT_LAPLACIAN_TEST(NAME) \
  TEST(CotangentLaplacian, NAME)           \
  {                                        \
    test(#NAME);                           \
  }

FLO_COTANGENT_LAPLACIAN_TEST(cube)
FLO_COTANGENT_LAPLACIAN_TEST(spot)

#undef FLO_COTANGENT_LAPLACIAN_TEST
