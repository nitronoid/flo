#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/similarity_xform.cuh"
#include <cusp/io/matrix_market.h>

namespace
{
using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;

void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Copy to device
  auto d_D = read_device_sparse_matrix<flo::real>(
    mp + "/intrinsic_dirac/intrinsic_dirac.mtx");

  DeviceVectorR d_xform(surf.n_vertices() * 4);
  flo::device::similarity_xform(d_D, d_xform, 1e-8, 3);
  HostVectorR h_xform = d_xform;

  // Read expected results
  auto expected_xform =
    read_host_dense_matrix<flo::real>(mp + "/similarity_xform/lambda.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_xform, Pointwise(FloatNear(0.001), expected_xform.values));
}
}  // namespace

#define FLO_SIMILARITY_XFORM_TEST(NAME) \
  TEST(SimilarityXform, NAME)           \
  {                                     \
    test(#NAME);                        \
  }

FLO_SIMILARITY_XFORM_TEST(cube)
FLO_SIMILARITY_XFORM_TEST(spot)

#undef FLO_SIMILARITY_XFORM_TEST
