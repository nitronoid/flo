#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/similarity_xform.cuh"
#include <cusp/transpose.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Copy to device
  auto d_D = read_device_sparse_matrix<flo::real>(
    mp + "/intrinsic_dirac/intrinsic_dirac.mtx");

  DeviceDenseMatrixR d_xform(4, surf.n_vertices());
  flo::device::similarity_xform(d_D, d_xform, 1e-12, 3);
  HostDenseMatrixR h_xform(d_xform.num_cols, d_xform.num_rows);
  cusp::transpose(d_xform, h_xform);

  // Read expected results
  auto expected_xform =
    read_host_dense_matrix<flo::real>(mp + "/similarity_xform/lambda.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_xform.values, Pointwise(FloatNear(0.001), expected_xform.values));
}
}  // namespace

#define FLO_SIMILARITY_XFORM_TEST(NAME) \
  TEST(SimilarityXform, NAME)           \
  {                                     \
    test(#NAME);                        \
  }

FLO_SIMILARITY_XFORM_TEST(cube)
FLO_SIMILARITY_XFORM_TEST(spot)
FLO_SIMILARITY_XFORM_TEST(bunny)

#undef FLO_SIMILARITY_XFORM_TEST
