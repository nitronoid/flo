#ifndef FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC
#define FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void cotangent_laplacian_values(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array2d<int, cusp::device_memory>::const_view di_entry_indices,
  cusp::array1d<int, cusp::device_memory>::const_view di_diagonals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian);

FLO_API void cotangent_laplacian(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_entry_indices,
  cusp::array1d<int, cusp::device_memory>::view do_diagonal_indices,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian);

FLO_API void cotangent_laplacian(
  cusp::array2d<real, cusp::device_memory>::const_view di_vertices,
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_cotangent_laplacian);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_COTANGENT_LAPLACIAN_ATOMIC
