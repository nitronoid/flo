#include "test_common.h"

#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/area.hpp"
#include "flo/host/valence.hpp"

#define INTRINSIC_DIRAC(NAME)                                                  \
  TEST(IntrinsicDirac, NAME)                                                   \
  {                                                                            \
    const std::string name = #NAME;                                            \
    const std::string matrix_prefix = "../matrices/" + name;                   \
    const auto& surf =                                                         \
      TestCache::get_mesh<TestCache::HOST>(name + ".obj");                     \
    std::vector<flo::real> rho(surf.n_vertices(), 3.0f);                       \
    Eigen::Matrix<int, 1, Eigen::Dynamic> dense_valence;                       \
    Eigen::loadMarketVector(                                                   \
      dense_valence, matrix_prefix + "/vertex_vertex_adjacency/valence.mtx");  \
    gsl::span<int> valence{dense_valence.data(),                               \
                           (size_t)dense_valence.size()};                      \
    Eigen::Matrix<flo::real, 1, Eigen::Dynamic> dense_area;                    \
    Eigen::loadMarketVector(dense_area, matrix_prefix + "/area/area.mtx");     \
    gsl::span<flo::real> face_area{dense_area.data(),                          \
                                   (size_t)dense_area.size()};                 \
    auto D = flo::host::intrinsic_dirac(                                       \
      surf.vertices, surf.faces, valence, face_area, rho);                     \
    Eigen::SparseMatrix<flo::real> expected_D;                                 \
    Eigen::loadMarket(expected_D,                                              \
                      matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx"); \
    EXPECT_MAT_NEAR(D, expected_D);                                            \
  }

INTRINSIC_DIRAC(cube)
INTRINSIC_DIRAC(spot)

#undef INTRINSIC_DIRAC

