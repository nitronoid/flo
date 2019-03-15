#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/area.cuh"

using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real4, cusp::device_memory>;
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
    SparseDeviceMatrix d_L(surf.n_vertices(),                                  \
                           surf.n_vertices(),                                  \
                           d_cumulative_valence.back() + surf.n_vertices());   \
    cusp::array1d<int, cusp::device_memory> d_diagonals(surf.n_vertices());    \
    flo::device::intrinsic_dirac(surf.vertices.data(),                         \
                                 surf.faces.data(),                            \
                                 d_area.data(),                                \
                                 d_rho.data(),                                 \
                                 d_cumulative_valence.data(),                  \
                                 d_offsets.data(),                             \
                                 surf.n_vertices(),                            \
                                 surf.n_faces(),                               \
                                 d_cumulative_valence.back(),                  \
                                 d_diagonals.data(),                           \
                                 d_L.row_indices.data(),                       \
                                 d_L.column_indices.data(),                    \
                                 d_L.values.data());                           \
    cusp::array1d<int, cusp::host_memory> h_diagonals = d_diagonals;           \
    cusp::array1d<int, cusp::host_memory> expected_diagonals;                  \
    cusp::io::read_matrix_market_file(expected_diagonals,                      \
                                      matrix_prefix +                          \
                                        "/cotangent_laplacian/diagonals.mtx"); \
    using namespace testing;                                                   \
    EXPECT_THAT(h_diagonals, Pointwise(Eq(), expected_diagonals));             \
  }

INTRINSIC_DIRAC_TEST(cube)
INTRINSIC_DIRAC_TEST(spot)
// INTRINSIC_DIRAC_TEST(dense_sphere_400x400)

