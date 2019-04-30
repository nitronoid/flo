#ifndef FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY
#define FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @breif Computes the vertex vertex adjacency with keys, the valence and the
/// cumulative valence per vertex
/// @param di_faces A column major matrix of face vertex indices
/// @param do_adjacency_keys An array containing the vertex vertex adjacency keys
/// @param do_adjacency  An array containing the vertex vertex adjacency 
/// @param do_valence  An array containing the vertex vertex valence
/// @param do_cumulative_valence An array containing the vertex vertex cumulative valence
/// @return the length of adjacency
FLO_API int vertex_vertex_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_VERTEX_VERTEX_ADJACENCY

