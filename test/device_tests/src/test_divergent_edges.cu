#include "test_common.h"
#include "device_test_util.h"
#include "flo/device/divergent_edges.cuh"
#include <cusp/transpose.h>

namespace
{
void test(std::string name)
{
  const std::string mp = "../matrices/" + name;
  auto& surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");

  // Copy to device
  auto d_xform =
    read_device_dense_matrix<flo::real>(mp + "/similarity_xform/lambda.mtx");
  auto d_L = read_device_sparse_matrix<flo::real>(
    mp + "/cotangent_laplacian/cotangent_laplacian.mtx");

  DeviceDenseMatrixR d_edges(4, d_xform.num_rows);

  // Run the function
  flo::device::divergent_edges(
    surf.vertices, surf.faces, d_xform.values, d_L, d_edges);
  HostDenseMatrixR h_edges = d_edges;

  // Read expected results
  auto expected_edge =
    read_host_dense_matrix<flo::real>(mp + "/divergent_edges/edges.mtx");

  auto expected_edges = expected_edge;
  cusp::transpose(expected_edge, expected_edges);

  // test our results
  using namespace testing;
  EXPECT_THAT(h_edges.values,
              Pointwise(FloatNear(FLOAT_SOFT_EPSILON), expected_edges.values));
}
}  // namespace

#define FLO_DIVERGENT_EDGES_TEST(NAME) \
  TEST(DivergentEdges, NAME)           \
  {                                    \
    test(#NAME);                       \
  }

FLO_DIVERGENT_EDGES_TEST(cube)
FLO_DIVERGENT_EDGES_TEST(spot)

#undef FLO_DIVERGENT_EDGES_TEST

