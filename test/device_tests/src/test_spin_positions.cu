#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/spin_positions.cuh"
#include "flo/device/intrinsic_dirac.cuh"
#include <cusp/transpose.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  auto d_ET =
    read_device_dense_matrix<flo::real>(mp + "/divergent_edges/edges.mtx");
  auto d_LQ = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/quaternion_cotangent_laplacian.mtx");

  cusp::array2d<flo::real, cusp::device_memory> d_E(d_ET.num_cols,
                                                    d_ET.num_rows);
  cusp::transpose(d_ET, d_E);
  cusp::print(d_E.values.subarray(0, 20));

  DeviceDenseMatrixR d_vertices(4, surf.n_vertices(), 0.f);
  flo::device::spin_positions(d_LQ, d_E, d_vertices);
  HostDenseMatrixR h_vertices = d_vertices;

  auto expected_vertices =
    read_host_dense_matrix<flo::real>(mp + "/spin_positions/positions.mtx");
  // test our results
  using namespace testing;
  EXPECT_THAT(h_vertices.values,
              Pointwise(FloatNear(0.00001), expected_vertices.values));
}
}  // namespace

#define FLO_SPIN_POSITIONS_TEST(NAME) \
  TEST(SpinPositions, NAME)           \
  {                                   \
    test(#NAME);                      \
  }

FLO_SPIN_POSITIONS_TEST(spot)
FLO_SPIN_POSITIONS_TEST(bunny)

#undef FLO_SPIN_POSITIONS_TEST

