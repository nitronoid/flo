#include "test_common.h"
#include "device_test_util.h"
#include <cusp/io/matrix_market.h>
#include "flo/device/vertex_mass.cuh"

namespace
{
void test(std::string name)
{
  const std::string matrix_prefix = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  cusp::array1d<int, cusp::host_memory> int_temp;
  cusp::array1d<flo::real, cusp::host_memory> real_temp;
  cusp::io::read_matrix_market_file(real_temp,
                                    matrix_prefix + "/area/area.mtx");
  cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;
  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/vertex_triangle_adjacency/valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_valence = int_temp;
  cusp::io::read_matrix_market_file(
    int_temp,
    matrix_prefix + "/vertex_triangle_adjacency/cumulative_valence.mtx");
  cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;
  cusp::io::read_matrix_market_file(
    int_temp, matrix_prefix + "/vertex_triangle_adjacency/adjacency.mtx");
  cusp::array1d<int, cusp::device_memory> d_adjacency = int_temp;

  cusp::array1d<flo::real, cusp::device_memory> d_vertex_mass(
    surf.n_vertices());
  flo::device::vertex_mass(
    d_area,
    d_adjacency,
    {d_cumulative_valence.begin() + 1, d_cumulative_valence.end()},
    d_vertex_mass);

  cusp::array1d<flo::real, cusp::host_memory> h_vertex_mass = d_vertex_mass;

  cusp::array1d<flo::real, cusp::host_memory> expected_mass;
  cusp::io::read_matrix_market_file(
    expected_mass, matrix_prefix + "/vertex_mass/vertex_mass.mtx");

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
