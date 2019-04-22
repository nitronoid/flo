#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"

namespace
{
using SparseDeviceMatrixQ =
  cusp::coo_matrix<int, flo::real4, cusp::device_memory>;
using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;
using flo::real;

template <typename T>
cusp::array1d<T, cusp::device_memory> read_device_vector(std::string path)
{
  cusp::array1d<T, cusp::host_memory> h_temp;
  cusp::array1d<T, cusp::device_memory> d_ret;
  cusp::io::read_matrix_market_file(h_temp, path);
  d_ret = h_temp;
  return d_ret;
}

void test(std::string name)
{
  // Set-up matrix path
  const std::string matrix_prefix = "../matrices/" + name;
  // Load the surface from our mesh cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Arbitrary constant rho
  cusp::array1d<flo::real, cusp::device_memory> d_rho(surf.n_vertices(), 3.f);

  // Read all our dependencies from disk
  auto d_area = read_device_vector<real>(matrix_prefix + "/area/area.mtx");
  auto d_valence = read_device_vector<int>(
    matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_adjacency = read_device_vector<int>(
    matrix_prefix + "/vertex_vertex_adjacency/adjacency.mtx");

  auto d_triangle_adjacency = read_device_vector<int>(
    matrix_prefix + "/vertex_triangle_adjacency/adjacency.mtx");
  auto d_triangle_cumulative_valence = read_device_vector<int>(
    matrix_prefix + "/vertex_triangle_adjacency/cumulative_valence.mtx");

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

  // Add an ascending sequence to the cumulative valence to account for
  // diagonals
  thrust::transform(d_cumulative_valence.begin() + 1,
                    d_cumulative_valence.end(),
                    thrust::make_counting_iterator(1),
                    d_cumulative_valence.begin() + 1,
                    thrust::plus<int>());

  // Allocate our real matrix for solving
  SparseDeviceMatrix d_Dr(
    surf.n_vertices() * 4, surf.n_vertices() * 4, d_Dq.values.size() * 16);

  // Transform our quaternion matrix to a real matrix
  flo::device::to_real_quaternion_matrix(
    d_Dq, {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()}, d_Dr);

  // Copy our results back to the host side
  SparseHostMatrix h_D(d_Dr.num_rows, d_Dr.num_cols, d_Dr.num_entries);
  h_D.values = d_Dr.values;
  h_D.row_indices = d_Dr.row_indices;
  h_D.column_indices = d_Dr.column_indices;
  cusp::array1d<int, cusp::host_memory> h_diagonals = d_diagonals;

  // Load our expected results from disk
  cusp::array1d<int, cusp::host_memory> expected_diagonals;
  cusp::io::read_matrix_market_file(
    expected_diagonals, matrix_prefix + "/cotangent_laplacian/diagonals.mtx");
  SparseHostMatrix expected_D;
  cusp::io::read_matrix_market_file(
    expected_D, matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_diagonals, Pointwise(Eq(), expected_diagonals));
  EXPECT_THAT(h_D.row_indices, Pointwise(Eq(), expected_D.row_indices));
  EXPECT_THAT(h_D.column_indices, Pointwise(Eq(), expected_D.column_indices));
  EXPECT_THAT(h_D.values,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_D.values));
}
}  // namespace

#define FLO_INTRINSIC_DIRAC_TEST(NAME) \
  TEST(IntrinsicDirac, NAME)           \
  {                                    \
    test(#NAME);                       \
  }

FLO_INTRINSIC_DIRAC_TEST(cube)
FLO_INTRINSIC_DIRAC_TEST(spot)

#undef FLO_INTRINSIC_DIRAC_TEST

