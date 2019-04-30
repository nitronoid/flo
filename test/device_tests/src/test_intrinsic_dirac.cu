#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"

namespace
{
void test(std::string name)
{
  // Set-up matrix path
  const std::string mp = "../matrices/" + name;
  // Load the surface from our mesh cache
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read all our dependencies from disk
  auto d_rho = read_device_vector<flo::real>(mp + "/project_basis/rho.mtx");
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

  // Allocate a sparse quaternion matrix to store our result
  DeviceSparseMatrixQ d_Dq(surf.n_vertices(),
                           surf.n_vertices(),
                           d_cumulative_valence.back() + surf.n_vertices());

  // Run our function
  flo::device::intrinsic_dirac(surf.vertices,
                               surf.faces,
                               d_adjacency_keys,
                               d_adjacency,
                               d_cumulative_valence,
                               d_area,
                               d_rho,
                               d_triangle_adjacency_keys,
                               d_triangle_adjacency,
                               d_Dq);

  // Add an ascending sequence to the cumulative valence to account for
  // diagonals
  thrust::transform(d_cumulative_valence.begin() + 1,
                    d_cumulative_valence.end(),
                    thrust::make_counting_iterator(1),
                    d_cumulative_valence.begin() + 1,
                    thrust::plus<int>());

  // Allocate our real matrix for solving
  DeviceSparseMatrixR d_Dr(
    surf.n_vertices() * 4, surf.n_vertices() * 4, d_Dq.values.size() * 16);
  // Transform our quaternion matrix to a real matrix
  flo::device::to_quaternion_matrix(
    d_Dq, {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()}, d_Dr);

  // Copy our results back to the host side
  HostSparseMatrixR h_D = d_Dr;

  // Load our expected results from disk
  auto expected_D = read_host_sparse_matrix<flo::real>(
    mp + "/intrinsic_dirac/intrinsic_dirac.mtx");
  auto expected_diagonals =
    read_host_vector<int>(mp + "/cotangent_laplacian/diagonals.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_D.row_indices, Pointwise(Eq(), expected_D.row_indices));
  EXPECT_THAT(h_D.column_indices, Pointwise(Eq(), expected_D.column_indices));
  EXPECT_THAT(h_D.values,
              Pointwise(FloatNear(0.005), expected_D.values));
}
}  // namespace

#define FLO_INTRINSIC_DIRAC_TEST(NAME) \
  TEST(IntrinsicDirac, NAME)           \
  {                                    \
    test(#NAME);                       \
  }

FLO_INTRINSIC_DIRAC_TEST(cube)
FLO_INTRINSIC_DIRAC_TEST(spot)
FLO_INTRINSIC_DIRAC_TEST(bunny)

#undef FLO_INTRINSIC_DIRAC_TEST

