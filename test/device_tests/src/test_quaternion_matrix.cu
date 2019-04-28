#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include "flo/device/intrinsic_dirac.cuh"

namespace
{
void test(std::string name)
{
  // Set-up matrix path
  const std::string mp = "../matrices/" + name;
  // Load the surface from our mesh cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read all our dependencies from disk
  auto d_L = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  // Add an ascending sequence to the cumulative valence to account for
  // diagonals
  thrust::transform(d_cumulative_valence.begin() + 1,
                    d_cumulative_valence.end(),
                    thrust::make_counting_iterator(1),
                    d_cumulative_valence.begin() + 1,
                    thrust::plus<int>());

  // Allocate our real matrix for solving
  DeviceSparseMatrixR d_QL(
    surf.n_vertices() * 4, surf.n_vertices() * 4, d_L.values.size() * 16);
  // Transform our quaternion matrix to a real matrix
  flo::device::to_real_quaternion_matrix(
    d_L, {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()}, d_QL);

  // Copy our results back to the host side
  HostSparseMatrixR h_QL = d_QL;

  // Load our expected results from disk
  auto expected_QL = read_host_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/quaternion_cotangent_laplacian.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_QL.row_indices, Pointwise(Eq(), expected_QL.row_indices));
  EXPECT_THAT(h_QL.column_indices, Pointwise(Eq(), expected_QL.column_indices));
  EXPECT_THAT(h_QL.values,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_QL.values));
}
}  // namespace

#define FLO_QUATERNION_MATRIX_TEST(NAME) \
  TEST(QuaternionMatrix, NAME)           \
  {                                      \
    test(#NAME);                         \
  }

FLO_QUATERNION_MATRIX_TEST(cube)
FLO_QUATERNION_MATRIX_TEST(spot)
FLO_QUATERNION_MATRIX_TEST(bunny)

#undef FLO_QUATERNION_MATRIX_TEST

