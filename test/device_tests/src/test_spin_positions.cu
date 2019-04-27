#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/spin_positions.cuh"
#include "flo/device/intrinsic_dirac.cuh"

namespace
{
struct LQZ
{
  int N;

  LQZ(int N) : N(N)
  {
  }

  __host__ __device__ flo::real
  operator()(const thrust::tuple<int, int, flo::real>& entry) const
  {
    const bool is_last_row_col =
      (entry.get<0>() == N - 1) || (entry.get<0>() == N - 2) ||
      (entry.get<0>() == N - 3) || (entry.get<0>() == N - 4) ||
      (entry.get<1>() == N - 1) || (entry.get<1>() == N - 2) ||
      (entry.get<1>() == N - 3) || (entry.get<1>() == N - 4);
    return is_last_row_col ? 0.f : entry.get<2>();
  }
};

void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  auto d_X =
    read_device_dense_matrix<flo::real>(mp + "/similarity_xform/lambda.mtx");
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
  printf("CWidth done\n");

  //// Convert to a quaternion matrix
  //DeviceSparseMatrixR d_LQ(
  //  surf.n_vertices() * 4, surf.n_vertices() * 4, d_L.num_entries * 16);
  //flo::device::to_real_quaternion_matrix(d_L, d_cv, d_LQ);
  //cusp::print(d_L);
  //printf("LQ done\n");
  //cusp::print(d_LQ);

  //auto entry_it =
  //  thrust::make_zip_iterator(thrust::make_tuple(d_LQ.column_indices.begin(),
  //                                               d_LQ.row_indices.begin(),
  //                                               d_LQ.values.begin()));

  //// Set final 4 rows and columns to zero
  //thrust::transform(entry_it,
  //                  entry_it + d_LQ.values.size(),
  //                  d_LQ.values.begin(),
  //                  LQZ(d_LQ.num_cols));
  //printf("LQ zero done\n");

  //// Set final xform to zero
  //thrust::copy_n(thrust::make_constant_iterator(0),
  //               4,
  //               thrust::make_permutation_iterator(
  //                 d_X.values.begin(),
  //                 thrust::make_transform_iterator(
  //                   thrust::make_counting_iterator(0),
  //                   [N = d_X.num_cols] __device__(int i) { return i * N; })));
  //printf("X zero done\n");

  //DeviceDenseMatrixR d_vertices(4, surf.n_vertices(), 0.f);
  //printf("Alloc V done\n");
  //flo::device::spin_positions(d_LQ, d_X, d_vertices);
  //printf("Spin done\n");
  //HostDenseMatrixR h_vertices = d_vertices;
  //printf("Host copy done\n");

  //auto expected_vertices =
  //  read_host_dense_matrix<flo::real>(mp + "/spin_positions/positions.mtx");
  //printf("Read done\n");
  //// test our results
  //using namespace testing;
  //EXPECT_THAT(h_vertices.values,
  //            Pointwise(FloatNear(0.001), expected_vertices.values));
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

