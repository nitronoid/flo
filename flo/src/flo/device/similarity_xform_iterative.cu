#include "flo/device/similarity_xform_iterative.cuh"
#include "flo/device/detail/diagonal_preconditioner.cuh"
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace iterative
{

FLO_API void similarity_xform(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac,
  cusp::array2d<real, cusp::device_memory>::view do_xform,
  const real i_tolerance,
  const int i_back_substitutions,
  const int i_max_convergence_iterations)
{
  cusp::array1d<real, cusp::device_memory> b(do_xform.num_entries);
  thrust::tabulate(b.begin(), b.end(), [] __device__(int x) {
    // When x is a multiple of 4, return one
    return !(x & 3);
  });

  cusp::monitor<flo::real> monitor(
      b, i_max_convergence_iterations, i_tolerance, 0.f, false);
  detail::DiagonalPreconditioner M(di_dirac);


  for (int iter = 0; iter < i_back_substitutions + 1; ++iter)
  {
    const real rnorm = 1.f / cusp::blas::nrm2(b);
    thrust::transform(b.begin(), b.end(), b.begin(), [=] __device__(real x) {
      return x * rnorm;
    });
    cusp::krylov::cg(di_dirac, do_xform.values, b, monitor, M);
    // Substitute back if we're not on the last iteration
    if (iter < i_back_substitutions + 1)
    {
      thrust::copy(do_xform.values.begin(), do_xform.values.end(), b.begin());
    }
  }

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
