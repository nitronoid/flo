#ifndef FLO_DEVICE_INCLUDED_SIMILARITY_XFORM_ITERATIVE
#define FLO_DEVICE_INCLUDED_SIMILARITY_XFORM_ITERATIVE

#include "flo/flo_internal.hpp"
#include "flo/device/cu_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

namespace iterative
{

FLO_API void similarity_xform(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_dirac,
  cusp::array2d<real, cusp::device_memory>::view do_xform,
  const real i_tolerance = 1e-7,
  const int i_back_substitutions = 0,
  const int i_max_convergence_iterations = 10000);

}

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_SIMILARITY_XFORM_ITERATIVE

