#include "flo/device/similarity_xform.cuh"
#include <thrust/tabulate.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/diagonal.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <cusp/permutation_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void similarity_xform(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac,
  cusp::array2d<real, cusp::device_memory>::view do_xform,
  const real i_tolerance,
  const int i_iterations)
{
  cu_raii::sparse::Handle sparse_handle;
  cu_raii::solver::SolverSp solver;

  similarity_xform(
    &sparse_handle, &solver, di_dirac, do_xform, i_tolerance, i_iterations);
}

FLO_API void similarity_xform(
  cu_raii::sparse::Handle* io_sparse_handle,
  cu_raii::solver::SolverSp* io_solver,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac,
  cusp::array2d<real, cusp::device_memory>::view do_xform,
  const real i_tolerance,
  const int i_iterations)
{
  // TODO: FIX this and ammend the tests
  // Convert the row indices to csr row offsets
  cusp::array1d<int, cusp::device_memory> row_offsets(di_dirac.num_rows + 1);
  cusp::indices_to_offsets(di_dirac.row_indices, row_offsets);

  // Fill our initial guess with the identity (quaternions)
  thrust::tabulate(
    do_xform.values.begin(), do_xform.values.end(), [] __device__(int x) {
      // When x is a multiple of 4, return one
      return !(x & 3);
    });
  cusp::array1d<real, cusp::device_memory> b(do_xform.num_entries);

  // Get a cuSolver and cuSparse handle
  io_solver->error_assert(__LINE__);
  io_sparse_handle->error_assert(__LINE__);

  // Create a matrix description
  cu_raii::sparse::MatrixDescription description_D(&io_sparse_handle->status);
  io_sparse_handle->error_assert(__LINE__);

  // Tell cuSparse what matrix to expect
  cusparseSetMatType(description_D, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(description_D, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(description_D, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatIndexBase(description_D, CUSPARSE_INDEX_BASE_ZERO);

  // Tell cusolver to use metis reordering
  const int reorder = 3;
  // cusolver will set this flag
  int singularity = 0;

  // Solve the system Dx = bx, using back substitution
  for (int iter = 0; iter < i_iterations + 1; ++iter)
  {
    // Get the norm of our right hand side
    const real rnorm = 1.f / cusp::blas::nrm2(do_xform.values);
    // Copy the result as our new right hand side, normalized
    thrust::transform(do_xform.values.begin(),
                      do_xform.values.end(),
                      b.begin(),
                      [=] __device__(real x) { return x * rnorm; });
    io_solver->status =
      cusolverSpScsrlsvchol(*io_solver,
                            di_dirac.num_rows,
                            di_dirac.num_entries,
                            description_D,
                            di_dirac.values.begin().base().get(),
                            row_offsets.data().get(),
                            di_dirac.column_indices.begin().base().get(),
                            b.begin().base().get(),
                            i_tolerance,
                            reorder,
                            do_xform.values.begin().base().get(),
                            &singularity);
    io_solver->error_assert(__LINE__);
  }
  if (singularity != -1)
    std::cout << "Singularity: " << singularity << '\n';

  // Normalize the result and re-arrange simultaneously to reduce kernel the
  // number of launches
  {
    // Get the norm of our right hand side
    const real rnorm = 1.f / cusp::blas::nrm2(do_xform.values);
    // We re-arrange to place W as the last component of each vector
    auto scatter_out = thrust::make_permutation_iterator(
      do_xform.values.begin(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [w = do_xform.num_cols] __device__(int i) {
          // Transpose our index, and
          // simultaneously shuffle in the order:
          // x -> w
          // y -> x
          // z -> y
          // w -> z
          const int32_t x = (i + 3) & 3;
          const int32_t y = i >> 2;
          return x * w + y;
        }));
    thrust::transform(do_xform.values.begin(),
                      do_xform.values.end(),
                      scatter_out,
                      [=] __device__(real x) { return x * rnorm; });
  }
}

FLO_DEVICE_NAMESPACE_END
