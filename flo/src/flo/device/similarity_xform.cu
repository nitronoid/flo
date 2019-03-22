#include "flo/device/similarity_xform.cuh"
#include <thrust/tabulate.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void similarity_xform(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac_matrix,
  cusp::array1d<real, cusp::device_memory>::view do_xform)
{
  thrust::tabulate(do_xform.begin(), do_xform.end(), [] __device__(int x) {
    // When x is a multiple of 4, return one
    return !(x & 3);
  });

  // Set-up stopping criteria
  cusp::monitor<real> monitor(do_xform, 100, 1e-7);

  for (int i = 0; i < 3; ++i)
  {
    // Conjugate gradient solve
    cusp::krylov::cg(di_dirac_matrix, do_xform, do_xform, monitor);
    // Normalize the result by first finding the norm of the result
    auto norm =
      thrust::transform_reduce(do_xform.begin(),
                               do_xform.end(),
                               [] __device__(real x) -> real { return x * x; },
                               0.f,
                               thrust::plus<real>());
    // We then divide through by the sqrt of the norm
    thrust::transform(do_xform.begin(),
                      do_xform.end(),
                      do_xform.begin(),
                      [inv_len = 1.f / std::sqrt(norm)] __device__(real x) {
                        return x * inv_len;
                      });
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
