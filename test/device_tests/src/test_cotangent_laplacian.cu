#include "test_common.h"
#include "device_test_util.h"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include "flo/device/cotangent_laplacian.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/vertex_mass.cuh"
#include "flo/device/area.cuh"

#include "flo/host/cotangent_laplacian.hpp"

using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;

TEST(CotangentLaplacianTriplets, cube)
{
  const std::string name = "cube";
  const std::string matrix_prefix = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  cusp::array1d<int, cusp::host_memory> int_temp;
  cusp::array1d<flo::real, cusp::host_memory> real_temp;
  cusp::io::read_matrix_market_file(real_temp,
                                    matrix_prefix + "/area/area.mtx");
  cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;
  cusp::io::read_matrix_market_file(
    int_temp,
    matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;

  auto d_L = flo::device::cotangent_laplacian(surf.vertices.data(),
                                              surf.faces.data(),
                                              d_area.data(),
                                              surf.n_vertices(),
                                              surf.n_faces(),
                                              d_cumulative_valence.back());
  SparseHostMatrix h_L;
  h_L.column_indices = d_L.column_indices;
  h_L.row_indices = d_L.row_indices;
  h_L.values = d_L.values;

  SparseHostMatrix expected_L;
  cusp::io::read_matrix_market_file(
    expected_L, matrix_prefix + "/cotangent_laplacian/cotangent_laplacian.mtx");

  using namespace testing;
  EXPECT_THAT(h_L.row_indices, Pointwise(Eq(), expected_L.row_indices));
  EXPECT_THAT(h_L.column_indices, Pointwise(Eq(), expected_L.column_indices));
  EXPECT_THAT(h_L.values,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_L.values));
}

#define COTANGENT_LAPLACIAN_ATOMIC_TEST(NAME)                                  \
  TEST(CotangentLaplacianAtomic, NAME)                                         \
  {                                                                            \
    const std::string name = #NAME;                                            \
    const std::string matrix_prefix = "../matrices/" + name;                   \
    const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");  \
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
    flo::device::cotangent_laplacian(surf.vertices.data(),                     \
                                     surf.faces.data(),                        \
                                     d_area.data(),                            \
                                     d_cumulative_valence.data(),              \
                                     d_offsets.data(),                         \
                                     surf.n_vertices(),                        \
                                     surf.n_faces(),                           \
                                     d_diagonals.data(),                       \
                                     d_L.row_indices.data(),                   \
                                     d_L.column_indices.data(),                \
                                     d_L.values.data());                       \
    SparseHostMatrix h_L;                                                      \
    h_L.column_indices = d_L.column_indices;                                   \
    h_L.row_indices = d_L.row_indices;                                         \
    h_L.values = d_L.values;                                                   \
    cusp::array1d<int, cusp::host_memory> h_diagonals = d_diagonals;           \
    SparseHostMatrix expected_L;                                               \
    cusp::io::read_matrix_market_file(                                         \
      expected_L,                                                              \
      matrix_prefix + "/cotangent_laplacian/cotangent_laplacian.mtx");         \
    cusp::array1d<int, cusp::host_memory> expected_diagonals;                  \
    cusp::io::read_matrix_market_file(expected_diagonals,                      \
                                      matrix_prefix +                          \
                                        "/cotangent_laplacian/diagonals.mtx"); \
    using namespace testing;                                                   \
    EXPECT_THAT(h_diagonals, Pointwise(Eq(), expected_diagonals));             \
    EXPECT_THAT(h_L.row_indices, Pointwise(Eq(), expected_L.row_indices));     \
    EXPECT_THAT(h_L.column_indices,                                            \
                Pointwise(Eq(), expected_L.column_indices));                   \
    EXPECT_THAT(h_L.values,                                                    \
                Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_L.values));  \
  }

COTANGENT_LAPLACIAN_ATOMIC_TEST(cube)
COTANGENT_LAPLACIAN_ATOMIC_TEST(spot)
// COTANGENT_LAPLACIAN_ATOMIC_TEST(dense_sphere_400x400)

// TEST(WRITE_MAT, mat)
//{
//  std::string name = "spot";
//  std::string matrix_prefix = "../matrices/" + name;
//  const auto& surf =
//    TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
//
//  cusp::array1d<int, cusp::device_memory> d_valence(surf.n_vertices());
//  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
//    surf.n_vertices() + 1);
//  cusp::array1d<int, cusp::device_memory> d_adjacency =
//    flo::device::vertex_vertex_adjacency(surf.faces.data(),
//                                         surf.n_faces(),
//                                         surf.n_vertices(),
//                                         d_valence.data(),
//                                         d_cumulative_valence.data());
//  std::cout << "Calc adjacency\n";
//  // cusp::io::write_matrix_market_file(
//  //  d_adjacency, matrix_prefix + "/vertex_vertex_adjacency/adjacency.mtx");
//  // std::cout<<"Write adjacency\n";
//  // cusp::io::write_matrix_market_file(
//  //  d_valence, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");
//  // std::cout<<"Write valence\n";
//  // cusp::io::write_matrix_market_file(
//  //  d_cumulative_valence,
//  //  matrix_prefix + "/vertex_vertex_adjacency/cumulative_valence.mtx");
//  // std::cout << "Write cumulative valence\n";
//
//  auto d_area =
//    flo::device::area(surf.vertices.data(), surf.faces.data(),
//    surf.n_faces());
//  auto d_offsets =
//    flo::device::adjacency_matrix_offset(surf.faces.data(),
//                                         d_adjacency.data(),
//                                         d_cumulative_valence.data(),
//                                         surf.n_faces());
//  std::cout << "Calc offsets\n";
//  thrust::device_ptr<int> optr{(int*)d_offsets.data().get()};
//  cusp::array1d<int, cusp::device_memory>::view offsets(
//    optr, optr + d_offsets.size() * 2);
//  //cusp::io::write_matrix_market_file(
//  //  offsets, matrix_prefix + "/adjacency_matrix_offset/offsets.mtx");
//  //std::cout << "Write offsets\n";
//  SparseDeviceMatrix d_L(surf.n_vertices(),
//                         surf.n_vertices(),
//                         d_cumulative_valence.back() + surf.n_vertices());
//  cusp::array1d<int, cusp::device_memory> d_diagonals(surf.n_vertices());
//  flo::device::cotangent_laplacian(surf.vertices.data(),
//                                   surf.faces.data(),
//                                   d_area.data(),
//                                   d_cumulative_valence.data(),
//                                   d_offsets.data(),
//                                   surf.n_vertices(),
//                                   surf.n_faces(),
//                                   d_cumulative_valence.back(),
//                                   d_diagonals.data(),
//                                   d_L.row_indices.data(),
//                                   d_L.column_indices.data(),
//                                   d_L.values.data());
//  cusp::io::write_matrix_market_file(
//    d_diagonals, matrix_prefix + "/cotangent_laplacian/diagonals.mtx");
//  std::cout << "Write diagonals\n";
//}

// TEST(REGRESSION, mat)
//{
//  std::string name = "spot";
//  std::string matrix_prefix = "../matrices/" + name;
//  const auto& d_surf =
//    TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
//
//  cusp::array1d<int, cusp::device_memory> d_valence(d_surf.n_vertices());
//  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
//    d_surf.n_vertices() + 1);
//  cusp::array1d<int, cusp::device_memory> d_adjacency =
//    flo::device::vertex_vertex_adjacency(d_surf.faces.data(),
//                                         d_surf.n_faces(),
//                                         d_surf.n_vertices(),
//                                         d_valence.data(),
//                                         d_cumulative_valence.data());
//  std::cout << d_surf.n_vertices() << '\n';
//  std::cout << d_cumulative_valence.back() + d_surf.n_vertices() << '\n';
//  int3 f0 = d_surf.faces[0];
//  printf("F0: %d, %d, %d,  CV: %d\n",
//         f0.x,
//         f0.y,
//         f0.z,
//         (int)d_cumulative_valence[f0.y]);
//
//  auto d_area = flo::device::area(
//    d_surf.vertices.data(), d_surf.faces.data(), d_surf.n_faces());
//  auto d_offsets =
//    flo::device::adjacency_matrix_offset(d_surf.faces.data(),
//                                         d_adjacency.data(),
//                                         d_cumulative_valence.data(),
//                                         d_surf.n_faces());
//  SparseDeviceMatrix d_L(d_surf.n_vertices(),
//                         d_surf.n_vertices(),
//                         d_cumulative_valence.back() + d_surf.n_vertices());
//  cusp::array1d<int, cusp::device_memory> d_diagonals(d_surf.n_vertices());
//  flo::device::cotangent_laplacian(d_surf.vertices.data(),
//                                   d_surf.faces.data(),
//                                   d_area.data(),
//                                   d_cumulative_valence.data(),
//                                   d_offsets.data(),
//                                   d_surf.n_vertices(),
//                                   d_surf.n_faces(),
//                                   d_cumulative_valence.back(),
//                                   d_diagonals.data(),
//                                   d_L.row_indices.data(),
//                                   d_L.column_indices.data(),
//                                   d_L.values.data());
//
//  cusp::array1d<int, cusp::host_memory> values = d_L.values;
//
//  const auto& h_surf =
//    TestCache::get_mesh<TestCache::HOST>(name + ".obj");
//  const auto h_L =
//    flo::host::cotangent_laplacian(h_surf.vertices, h_surf.faces);
//  gsl::span<const flo::real> expected_values{h_L.valuePtr(),
//                                             (uint)h_L.nonZeros()};
//  std::cout << expected_values.size() << '\n';
//
//  using namespace testing;
//  EXPECT_THAT(values,
//              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_values));
//}

//TEST(WRITE_MAT, mat)
//{
//  std::string name = "spot";
//  std::string matrix_prefix = "../matrices/" + name;
//  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
//  auto faces_copy = surf.faces;
//
//  cusp::array1d<flo::real, cusp::host_memory> real_temp;
//  cusp::io::read_matrix_market_file(real_temp,
//                                    matrix_prefix + "/area/area.mtx");
//  cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;
//
//  cusp::array1d<int, cusp::device_memory> d_adjacency(surf.n_faces() * 3);
//  cusp::array1d<int, cusp::device_memory> d_valence(surf.n_vertices());
//  cusp::array1d<int, cusp::device_memory> d_cumulative_valence(
//    surf.n_vertices() + 1);
//  flo::device::vertex_triangle_adjacency(
//    thrust::device_ptr<int>{(int*)faces_copy.data().get()},
//    surf.n_faces(),
//    surf.n_vertices(),
//    d_adjacency.data(),
//    d_valence.data(),
//    d_cumulative_valence.data());
//
//  cusp::array1d<flo::real, cusp::host_memory> d_mass = flo::device::vertex_mass(d_area.data(),
//                                         d_adjacency.data(),
//                                         d_valence.data(),
//                                         d_cumulative_valence.data(),
//                                         d_area.size(),
//                                         d_valence.size());
//  std::cout << "Calc adjacency\n";
//  cusp::io::write_matrix_market_file(
//    d_mass, matrix_prefix + "/vertex_mass/vertex_mass.mtx");
//  std::cout << "Write adjacency\n";
//}
