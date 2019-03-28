#include <benchmark/benchmark.h>
#include <numeric>
#include "test_common.h"
#include "flo/device/area.cuh"
#include "flo/device/intrinsic_dirac.cuh"
#include "flo/device/vertex_vertex_adjacency.cuh"
#include "flo/device/vertex_triangle_adjacency.cuh"
#include "flo/device/similarity_xform.cuh"
#include <cusp/coo_matrix.h>
#include <cusp/io/matrix_market.h>

namespace
{
using SparseHostMatrix = cusp::coo_matrix<int, flo::real, cusp::host_memory>;
using SparseDeviceMatrixQ =
  cusp::coo_matrix<int, flo::real4, cusp::device_memory>;
using SparseDeviceMatrix =
  cusp::coo_matrix<int, flo::real, cusp::device_memory>;

void bench_impl(std::string name, benchmark::State& state)
{
  const std::string matrix_prefix = "../../matrices/" + name;
  // Load our surface from the cache
  auto surf = TestCache::get_mesh<TestCache::DEVICE>(name + ".obj");
  // Read intrinsic dirac matrix
  SparseHostMatrix h_D;
  cusp::io::read_matrix_market_file(
    h_D, matrix_prefix + "/intrinsic_dirac/intrinsic_dirac.mtx");

  // Copy to device
  SparseDeviceMatrix d_Dr(h_D.num_rows, h_D.num_cols, h_D.values.size());
  d_Dr.row_indices = h_D.row_indices;
  d_Dr.column_indices = h_D.column_indices;
  d_Dr.values = h_D.values;

  cusp::array1d<flo::real, cusp::device_memory> d_xform(surf.n_vertices() * 4);
  for (auto _ : state)
  {
    flo::device::similarity_xform(d_Dr, d_xform);
  }
}
}  // namespace

#define FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(NAME)                   \
  static void DEVICE_similarity_xform_##NAME(benchmark::State& state) \
  {                                                                   \
    bench_impl(#NAME, state);                                         \
  }                                                                   \
  BENCHMARK(DEVICE_similarity_xform_##NAME);

FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(cube)
FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(spot)
FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(dense_sphere_400x400)
// FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK(dense_sphere_1000x1000)

#undef FLO_SIMILARITY_XFORM_DEVICE_BENCHMARK
