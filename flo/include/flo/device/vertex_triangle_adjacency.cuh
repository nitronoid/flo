#ifndef FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY
#define FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

#include "flo/flo_internal.hpp"
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @breif Computes the vertex triangle adjacency with keys, the valence and the
/// cumulative valence per vertex
/// @param di_faces A column major matrix of face vertex indices
/// @param do_adjacency_keys An array containing the vertex triangle adjacency keys
/// @param do_adjacency  An array containing the vertex triangle adjacency 
/// @param do_valence  An array containing the vertex triangle valence
/// @param do_cumulative_valence An array containing the vertex triangle cumulative valence
FLO_API void vertex_triangle_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_faces,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_valence,
  cusp::array1d<int, cusp::device_memory>::view do_cumulative_valence);

FLO_DEVICE_NAMESPACE_END

#endif  // FLO_DEVICE_INCLUDED_VERTEX_TRIANGLE_ADJACENCY

