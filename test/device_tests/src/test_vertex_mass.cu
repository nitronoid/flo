#include "test_common.h"
#include "device_test_util.h"

#include <cusp/io/matrix_market.h>
#include "flo/device/vertex_mass.cuh"

#define VERTEX_MASS_TEST(NAME)                                                \
  TEST(VertexMass, NAME)                                                      \
  {                                                                           \
    const std::string name = #NAME;                                           \
    const std::string matrix_prefix = "../matrices/" + name;                  \
    const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj"); \
    cusp::array1d<int, cusp::host_memory> int_temp;                           \
    cusp::array1d<flo::real, cusp::host_memory> real_temp;                    \
    cusp::io::read_matrix_market_file(real_temp,                              \
                                      matrix_prefix + "/area/area.mtx");      \
    cusp::array1d<flo::real, cusp::device_memory> d_area = real_temp;         \
    cusp::io::read_matrix_market_file(                                        \
      int_temp, matrix_prefix + "/vertex_triangle_adjacency/valence.mtx");    \
    cusp::array1d<int, cusp::device_memory> d_valence = int_temp;             \
    cusp::io::read_matrix_market_file(                                        \
      int_temp,                                                               \
      matrix_prefix + "/vertex_triangle_adjacency/cumulative_valence.mtx");   \
    cusp::array1d<int, cusp::device_memory> d_cumulative_valence = int_temp;  \
    cusp::io::read_matrix_market_file(                                        \
      int_temp, matrix_prefix + "/vertex_triangle_adjacency/adjacency.mtx");  \
    cusp::array1d<int, cusp::device_memory> d_adjacency = int_temp;           \
    auto d_mass = flo::device::vertex_mass(d_area.data(),                     \
                                           d_adjacency.data(),                \
                                           d_valence.data(),                  \
                                           d_cumulative_valence.data(),       \
                                           d_area.size(),                     \
                                           d_valence.size());                 \
    cusp::array1d<flo::real, cusp::host_memory> h_mass = d_mass;              \
    cusp::array1d<flo::real, cusp::host_memory> expected_mass;                \
    cusp::io::read_matrix_market_file(                                        \
      expected_mass, matrix_prefix + "/vertex_mass/vertex_mass.mtx");         \
    using namespace testing;                                                  \
    EXPECT_THAT(h_mass,                                                       \
                Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_mass));     \
  }

VERTEX_MASS_TEST(cube)
VERTEX_MASS_TEST(spot)
