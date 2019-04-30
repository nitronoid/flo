#include "flo/device/spin_positions_direct.cuh"
#include "flo/device/detail/diagonal_preconditioner.cuh"
#include "flo/device/detail/unary_functional.cuh"
#include "flo/device/detail/tuple_vector_conversion.cuh"
#include "flo/device/detail/quaternion_transpose_shuffle.cuh"
#include "flo/device/center_quaternions.cuh"
#include "flo/device/normalize_quaternions.cuh"

FLO_DEVICE_NAMESPACE_BEGIN

namespace direct
{

FLO_API void 
spin_positions(cusp::coo_matrix<int, real, cusp::device_memory>::const_view
                 di_quaternion_laplacian,
               cusp::array2d<real, cusp::device_memory>::const_view di_edges,
               cusp::array2d<real, cusp::device_memory>::view do_vertices,
               const real i_tolerance)
{
  cu_raii::sparse::Handle sparse_handle;
  cu_raii::solver::SolverSp solver;

  spin_positions(&sparse_handle,
                 &solver,
                 di_quaternion_laplacian,
                 di_edges,
                 do_vertices,
                 i_tolerance);
}

FLO_API void spin_positions(
  cu_raii::sparse::Handle* io_sparse_handle,
  cu_raii::solver::SolverSp* io_solver,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_quaternion_laplacian,
  cusp::array2d<real, cusp::device_memory>::const_view di_edges,
  cusp::array2d<real, cusp::device_memory>::view do_vertices,
  const real i_tolerance)
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

  // Get a cuSolver and cuSparse handle
  io_solver->error_assert(__LINE__);
  io_sparse_handle->error_assert(__LINE__);

  // Create a matrix description
  cu_raii::sparse::MatrixDescription description_QL(&io_sparse_handle->status);
  io_sparse_handle->error_assert(__LINE__);

  // Tell cuSparse what matrix to expect
  cusparseSetMatType(description_QL, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(description_QL, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(description_QL, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatIndexBase(description_QL, CUSPARSE_INDEX_BASE_ZERO);

#if __CUDACC_VER_MAJOR__ < 10
  // Tell cusolver to use symamd reordering if we're compiling with cuda 9
  const int reorder = 2;
#else
  // Tell cusolver to use metis reordering if we're compiling with cuda 10
  const int reorder = 3;
#endif
  // cusolver will set this flag
  int singularity = -1;


  io_solver->status = cusolverSpScsrlsvchol(
    *io_solver,
    di_quaternion_laplacian.num_rows,
    di_quaternion_laplacian.num_entries,
    description_QL,
    di_quaternion_laplacian.values.begin().base().get(),
    row_offsets.data().get(),
    di_quaternion_laplacian.column_indices.begin().base().get(),
    b.begin().base().get(),
    i_tolerance,
    reorder,
    do_vertices.values.begin().base().get(),
    &singularity);
  io_solver->error_assert(__LINE__);
  if (singularity != -1)
    std::cout << "Singularity: " << singularity << '\n';

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


