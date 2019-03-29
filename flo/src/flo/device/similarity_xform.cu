#include "flo/device/similarity_xform.cuh"
#include "flo/device/cu_raii.cuh"
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
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac_matrix,
  cusp::array1d<real, cusp::device_memory>::view do_xform,
  const real i_tolerance,
  const int i_back_substitution_iterations)
{
  // TODO: FIX this and ammend the tests
  // Convert the row indices to csr row offsets
  cusp::array1d<int, cusp::device_memory> row_offsets(di_dirac_matrix.num_rows +
                                                      1);
  cusp::indices_to_offsets(di_dirac_matrix.row_indices, row_offsets);

  // Fill our initial guess with the identity (quaternions)
  thrust::tabulate(do_xform.begin(), do_xform.end(), [] __device__(int x) {
    // When x is a multiple of 4, return one
    return !(x & 3);
  });

  // b is filled with ones then normalized
  cusp::array1d<real, cusp::device_memory> b(do_xform.size());

  // Get a cuSolver and cuSparse handle
  ScopedCuSolverSparse solver;
  // solver.error_assert(__LINE__);
  ScopedCuSparse cu_sparse;
  // cu_sparse.error_assert(__LINE__);

  // Create a matrix description
  ScopedCuSparseMatrixDescription description_D(&cu_sparse.status);
  // cu_sparse.error_assert(__LINE__);

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
  for (int iter = 0; iter < i_back_substitution_iterations + 1; ++iter)
  {
    cusp::blas::copy(do_xform, b);
    const real norm = cusp::blas::nrm2(b);
    cusp::blas::scal(b, 1.f / norm);
    solver.status =
      cusolverSpScsrlsvchol(solver,
                            di_dirac_matrix.num_rows,
                            di_dirac_matrix.num_entries,
                            description_D,
                            di_dirac_matrix.values.begin().base().get(),
                            row_offsets.data().get(),
                            di_dirac_matrix.column_indices.begin().base().get(),
                            b.begin().base().get(),
                            i_tolerance,
                            reorder,
                            do_xform.begin().base().get(),
                            &singularity);

    // solver.error_assert(__LINE__);
    // cudaDeviceSynchronize();
  }
  if (singularity != -1)
  std::cout << "Singularity:"<< singularity << '\n';

  // Normalize the result
  {
    const real norm = cusp::blas::nrm2(do_xform);
    cusp::blas::scal(do_xform, 1.f / norm);
  }

  // Re-arrange result to place W last
  thrust::scatter(
    do_xform.begin(),
    do_xform.end(),
    thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                    [] __device__(int x) {
                                      // Shuffle everything down by one, but if
                                      // you were an X component, add 4 to place
                                      // it as a W component
                                      return (x - 1) + !(x & 3) * 4;
                                    }),
    do_xform.begin());
}

FLO_DEVICE_NAMESPACE_END
