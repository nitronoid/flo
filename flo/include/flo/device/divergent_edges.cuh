#ifndef FLO_DEVICE_INCLUDED_DIVERGENT_EDGES
#define FLO_DEVICE_INCLUDED_DIVERGENT_EDGES

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void divergent_edges(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_xform,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_cotangent_laplacian,
  cusp::array2d<real, cusp::device_memory>::view do_edges);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_DIVERGENT_EDGES

