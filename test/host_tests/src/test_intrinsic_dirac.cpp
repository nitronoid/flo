#include "test_common.h"

#include "flo/host/intrinsic_dirac.hpp"
#include "flo/host/area.hpp"
#include "flo/host/valence.hpp"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::HOST>(name + ".obj");

  std::vector<flo::real> rho(surf.n_vertices(), 3.0f);
  auto vertex_valence =
    read_vector<int>(mp + "/vertex_vertex_adjacency/valence.mtx");
  auto face_area = read_vector<flo::real>(mp + "/face_area/face_area.mtx");

  auto D = flo::host::intrinsic_dirac(
    surf.vertices, surf.faces, vertex_valence, face_area, rho);

  auto expected_D =
    read_sparse_matrix<flo::real>(mp + "/intrinsic_dirac/intrinsic_dirac.mtx");

  EXPECT_MAT_NEAR(D, expected_D);
}
}  // namespace

#define FLO_INTRINSIC_DIRAC_TEST(NAME) \
  TEST(IntrinsicDirac, NAME)           \
  {                                    \
    test(#NAME);                       \
  }

FLO_INTRINSIC_DIRAC_TEST(cube)
FLO_INTRINSIC_DIRAC_TEST(spot)

#undef FLO_INTRINSIC_DIRAC_TEST

