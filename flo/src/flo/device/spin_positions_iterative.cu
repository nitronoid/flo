#include "flo/device/spin_positions_direct.cuh"
#include "flo/device/detail/diagonal_preconditioner.cuh"
#include "flo/device/detail/unary_functional.cuh"
#include "flo/device/detail/tuple_vector_conversion.cuh"
#include "flo/device/detail/quaternion_transpose_shuffle.cuh"
#include "flo/device/center_quaternions.cuh"
#include "flo/device/normalize_quaternions.cuh"
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace iterative
{

FLO_API void 
spin_positions(cusp::coo_matrix<int, real, cusp::device_memory>::const_view
                 di_quaternion_laplacian,
               cusp::array2d<real, cusp::device_memory>::const_view di_edges,
               cusp::array2d<real, cusp::device_memory>::view do_vertices,
               const real i_tolerance = 1e-7,
               const int i_max_convergence_iterations = 10000)
{
  // Convert the row indices to csr row offsets
  cusp::array1d<int, cusp::device_memory> row_offsets(
    di_quaternion_laplacian.num_rows + 1);
  cusp::indices_to_offsets(di_quaternion_laplacian.row_indices, row_offsets);

  cusp::array1d<real, cusp::device_memory> b(di_edges.num_entries);

  // Transpose the input edges, and simultaneously shuffle the W element to the start
  auto count_it = thrust::make_counting_iterator(0);
  auto shuffle_it = thrust::make_transform_iterator(
      count_it, detail::quaternion_transpose_shuffle(di_edges.num_cols));
  thrust::scatter(
    di_edges.values.begin(), di_edges.values.end(), shuffle_it, b.begin());

  cusp::monitor<flo::real> monitor(
      b, i_max_convergence_iterations, i_tolerance, 0.f, false);
  detail::DiagonalPreconditioner M(di_quaternion_laplacian);

    cusp::krylov::cg(di_quaternion_laplacian, do_vertices.values, b, monitor, M);
  {
    thrust::copy(
      do_vertices.values.begin(), do_vertices.values.end(), b.begin());
    auto xin_ptr =
      thrust::device_pointer_cast(reinterpret_cast<real4*>(b.data().get()));
    auto xout_ptr =
      thrust::make_zip_iterator(thrust::make_tuple(do_vertices.row(3).begin(),
                                                   do_vertices.row(0).begin(),
                                                   do_vertices.row(1).begin(),
                                                   do_vertices.row(2).begin()));

    thrust::transform(xin_ptr,
                      xin_ptr + do_vertices.num_cols,
                      xout_ptr,
                      detail::vector_to_tuple<real4>());

  }

  // Center and normalize the new positions
  center_quaternions(do_vertices);
  normalize_quaternions(do_vertices);
}

}

FLO_DEVICE_NAMESPACE_END

