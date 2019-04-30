#include "flo/device/similarity_xform_direct.cuh"
#include <thrust/tabulate.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/diagonal.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <cusp/print.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace direct
{

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
  // Convert the row indices to csr row offsets
  cusp::array1d<int, cusp::device_memory> row_offsets(di_dirac.num_rows + 1);
  cusp::indices_to_offsets(di_dirac.row_indices, row_offsets);

  // Fill our initial guess with the identity (quaternions)
  cusp::array1d<real, cusp::device_memory> b(di_dirac.num_cols);
  thrust::tabulate(
    do_xform.values.begin(), do_xform.values.end(), [] __device__(int x) {
      // When x is a multiple of 4, return one
      return !(x & 3);
    });

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

#if __CUDACC_VER_MAJOR__ < 10
  // Tell cusolver to use symamd reordering if we're compiling with cuda 9
  const int reorder = 2;
#else
  // Tell cusolver to use metis reordering if we're compiling with cuda 10
  const int reorder = 3;
#endif
  // cusolver will set this flag
  int singularity = -1;

  // Solve the system Dx = bx, using back substitution
  for (int iter = 0; iter < (i_iterations + 1) || singularity != -1; ++iter)
  {
    const real rnorm = 1.f / cusp::blas::nrm2(do_xform.values);
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
  {
    std::cout << "Singularity: " << singularity << '\n';
  }

  // Normalize the result and re-arrange simultaneously to reduce kernel the
  // number of launches
  {
    // Normalize and shuffle in the same kernel call
    const real rnorm = 1.f / cusp::blas::nrm2(do_xform.values);
    thrust::copy(do_xform.values.begin(), do_xform.values.end(), b.begin());
    auto xin_ptr = thrust::device_pointer_cast(
      reinterpret_cast<real4*>(b.data().get()));
    auto xout_ptr =
      thrust::make_zip_iterator(thrust::make_tuple(do_xform.row(3).begin(),
                                                   do_xform.row(0).begin(),
                                                   do_xform.row(1).begin(),
                                                   do_xform.row(2).begin()));

    thrust::transform(
      xin_ptr, xin_ptr + do_xform.num_cols, xout_ptr, 
      [=] __device__ (real4 quat)
      {
        return thrust::make_tuple(
          quat.x * rnorm, quat.y * rnorm, quat.z * rnorm, quat.w * rnorm);
      });
  }
}

}
FLO_DEVICE_NAMESPACE_END
