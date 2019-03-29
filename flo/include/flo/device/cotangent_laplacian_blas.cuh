#ifndef FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_BLAS
#define FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_BLAS

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void
edges(cusp::array1d<real, cusp::device_memory>::const_view di_vertices,
      cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
      cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
      cusp::array1d<real, cusp::device_memory>::view do_edges);

FLO_API void cotangent_laplacian(
  cusp::array1d<real, cusp::device_memory>::const_view di_edges,
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<int, cusp::device_memory>::view do_diagonals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view
    do_cotangent_laplacian);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_BLAS
