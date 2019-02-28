#include "test_common.h"
#include "flo/host/cotangent_laplacian.hpp"
#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/area.hpp"
#include "flo/host/valence.hpp"
#include "flo/host/flo_matrix_operation.hpp"

#define COTANGENT_LAPLACIAN_TEST(NAME)                                    \
  TEST(CotangentLaplacian, NAME)                                          \
  {                                                                       \
    const std::string name = #NAME;                                       \
    const std::string matrix_prefix = "../matrices/" + name;              \
    const auto& surf =                                                    \
      TestCache::get_mesh<TestCache::HOST>(name + ".obj");                \
    auto L = flo::host::cotangent_laplacian(surf.vertices, surf.faces);   \
    Eigen::SparseMatrix<flo::real> expected_L;                            \
    Eigen::loadMarket(expected_L,                                         \
                      matrix_prefix +                                     \
                        "/cotangent_laplacian/cotangent_laplacian.mtx");  \
    EXPECT_MAT_NEAR(L, expected_L);                                       \
  }

COTANGENT_LAPLACIAN_TEST(cube)
COTANGENT_LAPLACIAN_TEST(spot)
// COTANGENT_LAPLACIAN_TEST(dense_sphere_400x400)
// COTANGENT_LAPLACIAN_TEST(dense_sphere_1000x1000)
// COTANGENT_LAPLACIAN_TEST(dense_sphere_1500x1500)

#undef COTANGENT_LAPLACIAN_TEST

// TEST(WRITE, MAT)
//{
//  std::string name = "spot";
//  std::string matrix_prefix = "../matrices/" + name;
//  const auto& surf =
//    TestCache::get_mesh<TestCache::HOST>("../models/" + name + ".obj");
//
//  auto v = flo::host::valence(surf.faces);
//  auto V = flo::host::array_to_matrix(gsl::make_span(v));
//  Eigen::saveMarketVector(
//    V, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");
//
//  auto a = flo::host::area(surf.vertices, surf.faces);
//  auto A = flo::host::array_to_matrix(gsl::make_span(a));
//  Eigen::saveMarketVector(A, matrix_prefix + "/area/area.mtx");
//
//  // auto L = flo::host::cotangent_laplacian(surf.vertices, surf.faces);
//  // Eigen::saveMarket(
//  //  L, matrix_prefix + "/cotangent_laplacian/cotangent_laplacian.mtx");
//
//  // std::vector<flo::real> rho(surf.n_vertices(), 3.0f);
//  // auto D = flo::host::intrinsic_dirac(surf.vertices, surf.faces, v, a,
//  rho);
//  // Eigen::saveMarket(D, matrix_prefix +
//  // "/intrinsic_dirac/intrinsic_dirac.mtx");
//}

