#include "flo/device/similarity_xform.cuh"
#include <thrust/tabulate.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/diagonal.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <cusp/print.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace iterative
{

namespace
{
struct quat_shfl
{
  using tup4 = thrust::tuple<real, real, real, real>;

  __host__ __device__ tup4 operator()(real4 quat) const
  {
    return thrust::make_tuple(quat.y, quat.z, quat.w, quat.x);
  }
};

class diagonal_precond : public cusp::linear_operator<flo::real, cusp::device_memory>
{
  using Parent = cusp::linear_operator<flo::real, cusp::device_memory>;
  cusp::array1d<flo::real, cusp::device_memory> diagonal_reciprocals;

public:
  diagonal_precond(cusp::coo_matrix<int, flo::real, cusp::device_memory>::const_view di_A)
    : diagonal_reciprocals(di_A.num_rows)
  {
    // extract the main diagonal
    thrust::fill(diagonal_reciprocals.begin(), diagonal_reciprocals.end(), 0.f);
    thrust::scatter_if(di_A.values.begin(), di_A.values.end(),
                       di_A.row_indices.begin(),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(thrust::make_tuple(
                               di_A.row_indices.begin(), di_A.column_indices.begin())),
                           cusp::equal_pair_functor<int>()),
                       diagonal_reciprocals.begin());

    // invert the entries
    thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                      diagonal_reciprocals.begin(), cusp::reciprocal_functor<flo::real>());
  }

  template <typename VectorType1, typename VectorType2>
  void operator()(const VectorType1& x, VectorType2& y) const
  {
    cusp::blas::xmy(diagonal_reciprocals, x, y);
  }
};
}

FLO_API void similarity_xform(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac,
  cusp::array2d<real, cusp::device_memory>::view do_xform,
  const real i_tolerance,
  const int i_back_substitutions,
  const real i_max_convergence_iterations)
{
  cusp::array1d<real, cusp::device_memory> b(do_xform.num_entries);
  thrust::tabulate(b.begin(), b.end(), [] __device__(int x) {
    // When x is a multiple of 4, return one
    return !(x & 3);
  });

  cusp::monitor<flo::real> monitor(
      b, i_max_convergence_iterations, i_tolerance, 0.f, false);
  diagonal_precond M(di_dirac);


  for (int iter = 0; iter < i_iterations + 1; ++iter)
  {
    const real rnorm = 1.f / cusp::blas::nrm2(b);
    thrust::transform(b.begin(), b.end(), b.begin(), [=] __device__(real x) {
      return x * rnorm;
    });
    cusp::krylov::cg(di_dirac, do_xform.values, b, monitor);
    // Substitute back if we're not on the last iteration
    if (iter < i_iterations + 1)
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
      thrust::make_zip_iterator(thrust::make_tuple(do_xform.row(0).begin(),
                                                   do_xform.row(1).begin(),
                                                   do_xform.row(2).begin(),
                                                   do_xform.row(3).begin()));

    thrust::transform(
      xin_ptr, xin_ptr + do_xform.num_cols, xout_ptr, 
      [=] __device__ (real4 quat)
      {
        return thrust::make_tuple(
          quat.y * rnorm, quat.z * rnorm, quat.w * rnorm, quat.x * rnorm);
      });
  }
}

}
FLO_DEVICE_NAMESPACE_END
