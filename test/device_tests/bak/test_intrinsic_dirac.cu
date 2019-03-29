#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/area.cuh"

namespace
{
using SparseDeviceMatrixQ =
  cusp::coo_matrix<int, flo::real4, cusp::device_memory>;
using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;

void test(std::string name)
{
  // Set-up matrix path
  const std::string matrix_prefix = "../matrices/" + name;
  // Load the surface from our mesh cache
  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Arbitrary constant rho
  cusp::array1d<flo::real, cusp::device_memory> d_rho(surf.n_vertices(), 3.f);

  // Read all our dependencies from disk
  cusp::array1d<int, cusp::host_memory> int_temp;
  cusp::array1d<flo::real, cusp::host_memory> real_temp;
  cusp::io::read_matrix_market_file(real_temp,
                                    matrix_prefix + "/area/area.mtx");
  cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;
  cusp::io::read_matrix_market_file(
    int_temp,
    matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;
  cusp::array1d<int2, cusp::device_memory> d_offsets(surf.n_faces() * 3);
  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/adjacency_matrix_offset/offsets.mtx");
  thrust::copy_n((int2*)int_temp.data(), int_temp.size() / 2, d_offsets.data());
  cusp::io::read_matrix_market_file(
    int_temp,
    matrix_prefix + "/vertex_triangle_adjacency/cumulative_valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_cumulative_triangle_valence =
    int_temp;
  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/vertex_triangle_adjacency/adjacency.mtx");
  cusp::array1d<int, cusp::device_memory> d_triangle_adjacency = int_temp;

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
                               d_cumulative_triangle_valence,
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
  SparseHostMatrix h_D;
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

