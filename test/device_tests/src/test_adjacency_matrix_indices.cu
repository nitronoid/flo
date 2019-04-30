#include "test_common.h"
#include "device_test_util.h"
#include <cusp/io/matrix_market.h>
#include "flo/device/adjacency_matrix_indices.cuh"
#include <cusp/print.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read input matrices from disk
  auto d_adjacency_keys =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency_keys.mtx");
  auto d_adjacency =
    read_device_vector<int>(mp + "/vertex_vertex_adjacency/adjacency.mtx");
  auto d_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_vertex_adjacency/cumulative_valence.mtx");

  // Declare device side arrays to dump our results
  DeviceDenseMatrixI d_entry_indices(6, surf.n_faces());
  DeviceVectorI d_diagonal_indices(surf.n_vertices());
  DeviceSparseMatrixR d_L(surf.n_vertices(),
                          surf.n_vertices(),
                          d_cumulative_valence.back() + surf.n_vertices());

  auto temp_ptr = thrust::device_pointer_cast(
      reinterpret_cast<void*>(d_L.values.begin().base().get()));

  // Run the function
  flo::device::adjacency_matrix_indices(surf.faces,
                                        d_adjacency_keys,
                                        d_adjacency,
                                        d_cumulative_valence, 
                                        d_entry_indices,
                                        d_diagonal_indices,
                                        d_L.row_indices,
                                        d_L.column_indices,
                                        temp_ptr);

  HostDenseMatrixI h_entry_indices = d_entry_indices;
  HostVectorI h_diagonal_indices = d_diagonal_indices;
  HostSparseMatrixR h_L = d_L;

  auto expected_entry_indices =
    read_host_dense_matrix<int>(mp + "/adjacency_matrix_indices/indices.mtx");
  auto expected_diagonal_indices =
    read_host_vector<int>(mp + "/cotangent_laplacian/diagonals.mtx");
  // Load our expected results from disk
  auto expected_L = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");

  // Test the results
  using namespace testing;
  EXPECT_THAT(h_entry_indices.values, ElementsAreArray(expected_entry_indices.values));
  EXPECT_THAT(h_diagonal_indices, Pointwise(Eq(), expected_diagonal_indices));
  EXPECT_THAT(h_L.row_indices, Pointwise(Eq(), expected_L.row_indices));
  EXPECT_THAT(h_L.column_indices, Pointwise(Eq(), expected_L.column_indices));
}
}  // namespace

#define FLO_ADJACENCY_MATRIX_INDICES_TEST(NAME) \
  TEST(AdjacencyMatrixIndices, NAME)            \
  {                                             \
    test(#NAME);                                \
  }

FLO_ADJACENCY_MATRIX_INDICES_TEST(cube)
FLO_ADJACENCY_MATRIX_INDICES_TEST(spot)
FLO_ADJACENCY_MATRIX_INDICES_TEST(bunny)

#undef FLO_ADJACENCY_MATRIX_INDICES_TEST
