#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/area.cuh"

using SparseDeviceMatrixQ =
  cusp::coo_matrix<int, flo::real4, cusp::device_memory>;
using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;

#define INTRINSIC_DIRAC_TEST(NAME)                                             \
  TEST(IntrinsicDirac, NAME)                                                   \
  {                                                                            \
    const std::string name = #NAME;                                            \
    const std::string matrix_prefix = "../matrices/" + name;                   \
    const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");  \
    cusp::array1d<flo::real, cusp::device_memory> d_rho(surf.n_vertices(),     \
                                                        3.f);                  \
    cusp::array1d<int, cusp::host_memory> int_temp;                            \
    cusp::array1d<flo::real, cusp::host_memory> real_temp;                     \
    cusp::io::read_matrix_market_file(real_temp,                               \
                                      matrix_prefix + "/area/area.mtx");       \
    cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;          \
    cusp::io::read_matrix_market_file(                                         \
      int_temp, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");       \
    cusp::array1d<int, cusp::device_memory> d_valence = int_temp;              \
    cusp::io::read_matrix_market_file(                                         \
      int_temp,                                                                \
      matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");      \
    cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;   \
    cusp::io::read_matrix_market_file(                                         \
      int_temp, matrix_prefix + "/vertex_vertex_adjacency/adjacency.mtx");     \
    cusp::array1d<int, cusp::device_memory> d_adjacency = int_temp;            \
    cusp::array1d<int2, cusp::device_memory> d_offsets(surf.n_faces() * 3);    \
    cusp::io::read_matrix_market_file(                                         \
      int_temp, matrix_prefix + "/adjacency_matrix_offset/offsets.mtx");       \
    thrust::copy_n(                                                            \
      (int2*)int_temp.data(), int_temp.size() / 2, d_offsets.data());          \
    cusp::io::read_matrix_market_file(                                         \
      int_temp,                                                                \
      matrix_prefix + "/vertex_triangle_adjacency/cumulative_valence.mtx");    \
    cusp::array1d<int, cusp::device_memory> d_cumulative_triangle_valence =    \
      int_temp;                                                                \
    cusp::io::read_matrix_market_file(                                         \
      int_temp, matrix_prefix + "/vertex_triangle_adjacency/adjacency.mtx");   \
    cusp::array1d<int, cusp::device_memory> d_triangle_adjacency = int_temp;   \
    SparseDeviceMatrixQ d_Dq(surf.n_vertices(),                                \
                             surf.n_vertices(),                                \
                             d_cumulative_valence.back() + surf.n_vertices()); \
    cusp::array1d<int, cusp::device_memory> d_diagonals(surf.n_vertices());    \
    flo::device::intrinsic_dirac(surf.vertices.data(),                         \
                                 surf.faces.data(),                            \
                                 d_area.data(),                                \
                                 d_rho.data(),                                 \
                                 d_cumulative_valence.data(),                  \
                                 d_offsets.data(),                             \
                                 d_cumulative_triangle_valence.data(),         \
                                 d_triangle_adjacency.data(),                  \
                                 surf.n_vertices(),                            \
                                 surf.n_faces(),                               \
                                 d_diagonals.data(),                           \
                                 d_Dq.row_indices.data(),                      \
                                 d_Dq.column_indices.data(),                   \
                                 d_Dq.values.data());                          \
    thrust::transform(d_cumulative_valence.begin() + 1,                        \
                      d_cumulative_valence.end(),                              \
                      thrust::make_counting_iterator(1),                       \
                      d_cumulative_valence.begin() + 1,                        \
                      thrust::plus<int>());                                    \
                                                                               \
    SparseDeviceMatrix d_Dr(                                                   \
      surf.n_vertices() * 4, surf.n_vertices() * 4, d_Dq.values.size() * 16);  \
    flo::device::to_real_quaternion_matrix(d_Dq.row_indices.data(),            \
                                           d_Dq.column_indices.data(),         \
                                           d_Dq.values.data(),                 \
                                           d_cumulative_valence.data() + 1,    \
                                           d_Dq.values.size(),                 \
                                           d_Dr.row_indices.data(),            \
                                           d_Dr.column_indices.data(),         \
                                           d_Dr.values.data());                \
    SparseHostMatrix h_D;                                                      \
    h_D.values = d_Dr.values;                                                  \
    h_D.row_indices = d_Dr.row_indices;                                        \
    h_D.column_indices = d_Dr.column_indices;                                  \
    cusp::array1d<int, cusp::host_memory> h_diagonals = d_diagonals;           \
    cusp::array1d<int, cusp::host_memory> expected_diagonals;                  \
    cusp::io::read_matrix_market_file(expected_diagonals,                      \
                                      matrix_prefix +                          \
                                        "/cotangent_laplacian/diagonals.mtx"); \
    SparseHostMatrix expected_D;                                               \
    cusp::io::read_matrix_market_file(                                         \
      expected_D, matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx");     \
    using namespace testing;                                                   \
    EXPECT_THAT(h_diagonals, Pointwise(Eq(), expected_diagonals));             \
    EXPECT_THAT(h_D.row_indices, Pointwise(Eq(), expected_D.row_indices));     \
    EXPECT_THAT(h_D.column_indices,                                            \
                Pointwise(Eq(), expected_D.column_indices));                   \
    EXPECT_THAT(h_D.values,                                                    \
                Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_D.values));  \
  }

INTRINSIC_DIRAC_TEST(cube)
INTRINSIC_DIRAC_TEST(spot)
// INTRINSIC_DIRAC_TEST(dense_sphere_400x400)

