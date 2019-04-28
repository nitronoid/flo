#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/spin_positions.cuh"
#include "flo/device/intrinsic_dirac.cuh"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  auto d_E =
    read_device_dense_matrix<flo::real>(mp + "/divergent_edges/edges.mtx");
  auto d_L = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");
  auto d_cv = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  auto d_v =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  // Add an ascending sequence to the cumulative valence to account for
  // diagonals
  auto count = thrust::make_counting_iterator(1);
  thrust::transform(
    d_cv.begin() + 1, d_cv.end(), count, d_cv.begin() + 1, thrust::plus<int>());

  // Convert to a quaternion matrix
  DeviceSparseMatrixR d_LQ(
    surf.n_vertices() * 4, surf.n_vertices() * 4, d_L.num_entries * 16);
  flo::device::to_real_quaternion_matrix(
    d_L, {d_cv.begin() + 1, d_cv.end()}, d_LQ);

  auto entry_it =
    thrust::make_zip_iterator(thrust::make_tuple(d_LQ.column_indices.begin(),
                                                 d_LQ.row_indices.begin(),
                                                 d_LQ.values.begin()));

  DeviceDenseMatrixR d_vertices(4, surf.n_vertices(), 0.f);
  flo::device::spin_positions(d_LQ, d_E, d_vertices);
  HostDenseMatrixR h_vertices = d_vertices;

  auto expected_vertices =
    read_host_dense_matrix<flo::real>(mp + "/spin_positions/positions.mtx");
  // test our results
  using namespace testing;
  EXPECT_THAT(h_vertices.values,
              Pointwise(FloatNear(0.001), expected_vertices.values));
}
}  // namespace

#define FLO_SPIN_POSITIONS_TEST(NAME) \
  TEST(SpinPositions, NAME)           \
  {                                   \
    test(#NAME);                      \
  }

FLO_SPIN_POSITIONS_TEST(cube)
FLO_SPIN_POSITIONS_TEST(spot)
FLO_SPIN_POSITIONS_TEST(bunny)

#undef FLO_SPIN_POSITIONS_TEST

