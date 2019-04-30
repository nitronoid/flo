#ifndef FLO_DEVICE_INCLUDED_DIVERGENT_EDGES
#define FLO_DEVICE_INCLUDED_DIVERGENT_EDGES

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @breif Computes new edges that best fit the provided similarity transforms.
///  @param di_vertices A row major matrix of vertex positions
///  @param di_faces A row major matrix of face vertex indices
///  @param di_xform A row major matrix of quaternion transforms
///  @param di_cotangent_laplacian Sparse cotangent laplacian matrix
///  @param do_edges A row major matrix of best fit edges
FLO_API void divergent_edges(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<real, cusp::device_memory>::const_view di_xform,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_cotangent_laplacian,
  cusp::array2d<real, cusp::device_memory>::view do_edges);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_DIVERGENT_EDGES

