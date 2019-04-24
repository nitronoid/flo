#include "test_common.h"
#include "device_test_util.h"
#include <cusp/io/matrix_market.h>
#include "flo/device/vertex_mass.cuh"

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  auto d_area = read_device_vector<flo::real>(mp + "/face_area/face_area.mtx");
  auto d_triangle_cumulative_valence = read_device_vector<int>(
    mp + "/vertex_triangle_adjacency/cumulative_valence.mtx");
  auto d_triangle_adjacency =
    read_device_vector<int>(mp + "/vertex_triangle_adjacency/adjacency.mtx");

  DeviceVectorR d_vertex_mass(surf.n_vertices());

  flo::device::vertex_mass(d_area,
                           d_triangle_adjacency,
                           {d_triangle_cumulative_valence.begin() + 1,
                            d_triangle_cumulative_valence.end()},
                           d_vertex_mass);

  HostVectorR h_vertex_mass = d_vertex_mass;

  auto expected_mass =
    read_device_vector<flo::real>(mp + "/vertex_mass/vertex_mass.mtx");

  using namespace testing;
  EXPECT_THAT(h_vertex_mass,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_mass));
}
}  // namespace

#define FLO_VERTEX_MASS_TEST(NAME) \
  TEST(VertexMass, NAME)           \
  {                                \
    test(#NAME);                   \
  }

FLO_VERTEX_MASS_TEST(cube)
FLO_VERTEX_MASS_TEST(spot)

#undef FLO_VERTEX_MASS_TEST
