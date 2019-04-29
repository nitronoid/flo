#ifndef FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY
#define FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

// Returns the length of adjacency
FLO_API int vertex_vertex_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY

