#ifndef FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES
#define FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

FLO_DEVICE_NAMESPACE_BEGIN

FLO_API void adjacency_matrix_indices(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array2d<int, cusp::device_memory>::view do_indices);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_ADJACENCY_MATRIX_INDICES


