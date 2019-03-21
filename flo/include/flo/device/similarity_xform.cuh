#ifndef FLO_DEVICE_INCLUDED_SIMILARITY_XFORM
#define FLO_DEVICE_INCLUDED_SIMILARITY_XFORM

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void similarity_xform(
  const cusp::coo_matrix_view<
    typename cusp::array1d<int, cusp::device_memory>::const_view,
    typename cusp::array1d<int, cusp::device_memory>::const_view,
    typename cusp::array1d<real, cusp::device_memory>::const_view> di_dirac_matrix,
  cusp::array1d<real, cusp::device_memory>::view do_xform);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_SIMILARITY_XFORM
