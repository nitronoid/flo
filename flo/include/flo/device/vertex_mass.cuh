#ifndef FLO_DEVICE_INCLUDED_VERTEX_MASS
#define FLO_DEVICE_INCLUDED_VERTEX_MASS

#include "flo/flo_internal.hpp"
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

FLO_DEVICE_NAMESPACE_BEGIN

/// @breif Computes the vertex masses
/// @param di_face_area the per face areas
/// @param di_adjacency the vertex vertex adjacency
/// @param di_cumulative_valence the vertex vertex cumulative valence
/// @param do_vertex_mass the per vertex masses
FLO_API void vertex_mass(
  cusp::array1d<real, cusp::device_memory>::const_view di_face_area,
  cusp::array1d<int, cusp::device_memory>::const_view di_adjacency,
  cusp::array1d<int, cusp::device_memory>::const_view di_cumulative_valence,
  cusp::array1d<real, cusp::device_memory>::view do_vertex_mass);

FLO_DEVICE_NAMESPACE_END

#endif//FLO_DEVICE_INCLUDED_VERTEX_MASS

