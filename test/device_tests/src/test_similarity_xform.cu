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
  const std::string matrix_prefix = "../matrices/" + name;
  const auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Read intrinsic dirac matrix
  SparseHostMatrix h_D;
  cusp::io::read_matrix_market_file(
    h_D, matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx");

  // Copy to device
  SparseDeviceMatrix d_D(h_D.num_rows, h_D.num_cols, h_D.values.size());
  d_D.row_indices = h_D.row_indices;
  d_D.column_indices = h_D.column_indices;
  d_D.values = h_D.values;

  cusp::array1d<flo::real, cusp::device_memory> d_xform(surf.n_vertices() * 4);
  flo::device::similarity_xform(d_D, d_xform, 1e-8, 3);

  cusp::array1d<flo::real, cusp::host_memory> h_xform = d_xform;

  // flo::real4 q;
  // q.x = d_xform[0];
  // q.y = d_xform[1];
  // q.z = d_xform[2];
  // q.w = d_xform[3];
  // printf("Q: (%f, [%f, %f, %f])\n", q.w, q.x, q.y, q.z);

  // Read expected results
  cusp::array1d<flo::real, cusp::host_memory> expected_xform;
  cusp::io::read_matrix_market_file(
    expected_xform, matrix_prefix + "/similarity_xform/lambda.mtx");

  // test our results
  using namespace testing;
  EXPECT_THAT(h_xform, Pointwise(FloatNear(0.001), expected_xform));
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
